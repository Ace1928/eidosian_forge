import abc
import collections
import math
import numpy as np
import six
from tensorflow.python.eager import context
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import template
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
class _BiasLayer(base.Layer):
    """A layer for the bias term."""

    def __init__(self, units=1, trainable=True, weight_collections=None, name=None, **kwargs):
        super(_BiasLayer, self).__init__(trainable=trainable, name=name, **kwargs)
        self._units = units
        self._weight_collections = weight_collections

    def build(self, _):
        self._bias_variable = self.add_variable('bias_weights', shape=[self._units], initializer=init_ops.zeros_initializer(), trainable=self.trainable)
        _add_to_collections(self._bias_variable, self._weight_collections)
        self.built = True

    def call(self, _):
        return self._bias_variable