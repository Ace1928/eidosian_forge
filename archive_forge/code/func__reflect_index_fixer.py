import functools
import itertools
import operator
import tensorflow as tf
from keras.src.backend.tensorflow.core import convert_to_tensor
def _reflect_index_fixer(index, size):
    return tf.math.floordiv(_mirror_index_fixer(2 * index + 1, 2 * size + 1) - 1, 2)