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
class SmallSubclassMLP(models.Model):
    """A subclass model based small MLP."""

    def __init__(self, num_hidden, num_classes, use_bn=False, use_dp=False, **kwargs):
        super(SmallSubclassMLP, self).__init__(name='test_model', **kwargs)
        self.use_bn = use_bn
        self.use_dp = use_dp
        self.layer_a = layers.Dense(num_hidden, activation='relu')
        activation = 'sigmoid' if num_classes == 1 else 'softmax'
        self.layer_b = layers.Dense(num_classes, activation=activation)
        if self.use_dp:
            self.dp = layers.Dropout(0.5)
        if self.use_bn:
            self.bn = layers.BatchNormalization(axis=-1)

    def call(self, inputs, **kwargs):
        x = self.layer_a(inputs)
        if self.use_dp:
            x = self.dp(x)
        if self.use_bn:
            x = self.bn(x)
        return self.layer_b(x)