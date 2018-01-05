import os
import numpy as np
from PIL import Image, ImageFilter
import tensorflow as tf

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = script_dir + '/mnist/data/'
model_path = script_dir + '/models/mnist-cnn'



from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(data_path, one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1)),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1)),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.truncated_normal([1024, n_classes], stddev=0.1))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}




# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables



def load_image(filename):
    img = Image.open(filename).convert('L')

    # resize to 28x28
    img = img.resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)

    # normalization : 255 RGB -> 0, 1
    data = [(255 - x) * 1.0 / 255.0 for x in list(img.getdata())]

    # reshape -> [-1, 28, 28, 1]
    return np.reshape(data, (1,784));

def classify(feed_dict):
    number = sess.run(tf.argmax(pred, 1), feed_dict)[0]
    accuracy = sess.run(tf.nn.softmax(pred), feed_dict)[0]

    return number, accuracy[number]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess,model_path)
"""
print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:1],
                                      y: mnist.test.labels[:1],
                                      keep_prob: 1.}))

print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:2],
                                      y: mnist.test.labels[:2],
                                      keep_prob: 1.}))
"""
filename = script_dir+'/7.png'
data = load_image(filename)

number1 = sess.run(tf.argmax(pred, 1), feed_dict={x :data,keep_prob: 1.})[0]
accuracy1 = sess.run(tf.nn.softmax(pred), feed_dict={x :data,keep_prob: 1.})[0]



print("Accuracy result : ")
print(accuracy1)
print("Classified : ")
print(number1)