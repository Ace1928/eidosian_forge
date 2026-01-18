import collections
import pickle
import threading
import time
import timeit
from absl import flags
from absl import logging
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.distribute import values as values_lib  
from tensorflow.python.framework import composite_tensor  
from tensorflow.python.framework import tensor_conversion_registry  
class StructuredFIFOQueue(tf.queue.FIFOQueue):
    """A tf.queue.FIFOQueue that supports nests and tf.TensorSpec."""

    def __init__(self, capacity, specs, shared_name=None, name='structured_fifo_queue'):
        self._specs = specs
        self._flattened_specs = tf.nest.flatten(specs)
        dtypes = [ts.dtype for ts in self._flattened_specs]
        shapes = [ts.shape for ts in self._flattened_specs]
        super(StructuredFIFOQueue, self).__init__(capacity, dtypes, shapes)

    def dequeue(self, name=None):
        result = super(StructuredFIFOQueue, self).dequeue(name=name)
        return tf.nest.pack_sequence_as(self._specs, result)

    def dequeue_many(self, batch_size, name=None):
        result = super(StructuredFIFOQueue, self).dequeue_many(batch_size, name=name)
        return tf.nest.pack_sequence_as(self._specs, result)

    def enqueue(self, vals, name=None):
        tf.nest.assert_same_structure(vals, self._specs)
        return super(StructuredFIFOQueue, self).enqueue(tf.nest.flatten(vals), name=name)

    def enqueue_many(self, vals, name=None):
        tf.nest.assert_same_structure(vals, self._specs)
        return super(StructuredFIFOQueue, self).enqueue_many(tf.nest.flatten(vals), name=name)