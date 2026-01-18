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
class TPUEncodedF32Spec(tf.TypeSpec):
    """Type specification for composite tensor TPUEncodedF32Spec."""

    def __init__(self, encoded_shape, original_shape):
        self._value_specs = (tf.TensorSpec(encoded_shape, tf.float32),)
        self.original_shape = original_shape

    @property
    def _component_specs(self):
        return self._value_specs

    def _to_components(self, value):
        return (value.encoded,)

    def _from_components(self, components):
        return TPUEncodedF32(components[0], self.original_shape)

    def _serialize(self):
        return (self._value_specs[0].shape, self.original_shape)

    def _to_legacy_output_types(self):
        return self._value_specs[0].dtype

    def _to_legacy_output_shapes(self):
        return self._value_specs[0].shape

    @property
    def value_type(self):
        return TPUEncodedF32