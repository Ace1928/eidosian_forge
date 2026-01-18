from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
from tensorflow.python.util import function_utils
class DistributedIteratorInitializerHook(tf.compat.v1.train.SessionRunHook):
    """Creates a SessionRunHook that initializes the passed iterator."""

    def __init__(self, iterator):
        self._iterator = iterator

    def begin(self):
        self._initializer = self._iterator.initialize()

    def after_create_session(self, session, coord):
        del coord
        session.run(self._initializer)