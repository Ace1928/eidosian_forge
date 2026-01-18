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
class _NumericColumn(_DenseColumn, collections.namedtuple('_NumericColumn', ['key', 'shape', 'default_value', 'dtype', 'normalizer_fn'])):
    """see `numeric_column`."""

    @property
    def name(self):
        return self.key

    @property
    def _parse_example_spec(self):
        return {self.key: parsing_ops.FixedLenFeature(self.shape, self.dtype, self.default_value)}

    def _transform_feature(self, inputs):
        input_tensor = inputs.get(self.key)
        if isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
            raise ValueError('The corresponding Tensor of numerical column must be a Tensor. SparseTensor is not supported. key: {}'.format(self.key))
        if self.normalizer_fn is not None:
            input_tensor = self.normalizer_fn(input_tensor)
        return math_ops.cast(input_tensor, dtypes.float32)

    @property
    def _variable_shape(self):
        return tensor_shape.TensorShape(self.shape)

    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        """Returns dense `Tensor` representing numeric feature.

    Args:
      inputs: A `_LazyBuilder` object to access inputs.
      weight_collections: Unused `weight_collections` since no variables are
        created in this function.
      trainable: Unused `trainable` bool since no variables are created in this
        function.

    Returns:
      Dense `Tensor` created within `_transform_feature`.
    """
        del weight_collections
        del trainable
        return inputs.get(self)