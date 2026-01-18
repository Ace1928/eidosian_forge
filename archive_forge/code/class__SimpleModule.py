import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.distribute import model_collection_base
from keras.src.optimizers.legacy import gradient_descent
class _SimpleModule(tf.Module):

    def __init__(self):
        self.v = tf.Variable(3.0)

    @tf.function
    def __call__(self, x):
        return self.v * x