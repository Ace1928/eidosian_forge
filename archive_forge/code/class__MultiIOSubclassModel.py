import collections
import contextlib
import functools
import itertools
import threading
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import adadelta as adadelta_v2
from tensorflow.python.keras.optimizer_v2 import adagrad as adagrad_v2
from tensorflow.python.keras.optimizer_v2 import adam as adam_v2
from tensorflow.python.keras.optimizer_v2 import adamax as adamax_v2
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.keras.optimizer_v2 import nadam as nadam_v2
from tensorflow.python.keras.optimizer_v2 import rmsprop as rmsprop_v2
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util import tf_decorator
class _MultiIOSubclassModel(models.Model):
    """Multi IO Keras subclass model."""

    def __init__(self, branch_a, branch_b, shared_input_branch=None, shared_output_branch=None, name=None):
        super(_MultiIOSubclassModel, self).__init__(name=name)
        self._shared_input_branch = shared_input_branch
        self._branch_a = branch_a
        self._branch_b = branch_b
        self._shared_output_branch = shared_output_branch

    def call(self, inputs, **kwargs):
        if self._shared_input_branch:
            for layer in self._shared_input_branch:
                inputs = layer(inputs)
            a = inputs
            b = inputs
        elif isinstance(inputs, dict):
            a = inputs['input_1']
            b = inputs['input_2']
        else:
            a, b = inputs
        for layer in self._branch_a:
            a = layer(a)
        for layer in self._branch_b:
            b = layer(b)
        outs = [a, b]
        if self._shared_output_branch:
            for layer in self._shared_output_branch:
                outs = layer(outs)
        return outs