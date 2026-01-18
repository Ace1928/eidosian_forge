import functools
import itertools
import operator
import tensorflow as tf
from keras.src.backend.tensorflow.core import convert_to_tensor
def _mirror_index_fixer(index, size):
    s = size - 1
    return tf.abs((index + s) % (2 * s) - s)