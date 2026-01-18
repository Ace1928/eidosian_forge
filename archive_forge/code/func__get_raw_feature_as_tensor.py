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
def _get_raw_feature_as_tensor(self, key):
    """Gets the raw_feature (keyed by `key`) as `tensor`.

    The raw feature is converted to (sparse) tensor and maybe expand dim.

    For both `Tensor` and `SparseTensor`, the rank will be expanded (to 2) if
    the rank is 1. This supports dynamic rank also. For rank 0 raw feature, will
    error out as it is not supported.

    Args:
      key: A `str` key to access the raw feature.

    Returns:
      A `Tensor` or `SparseTensor`.

    Raises:
      ValueError: if the raw feature has rank 0.
    """
    raw_feature = self._features[key]
    feature_tensor = sparse_tensor_lib.convert_to_tensor_or_sparse_tensor(raw_feature)

    def expand_dims(input_tensor):
        if isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
            return sparse_ops.sparse_reshape(input_tensor, [array_ops.shape(input_tensor)[0], 1])
        else:
            return array_ops.expand_dims(input_tensor, -1)
    rank = feature_tensor.get_shape().ndims
    if rank is not None:
        if rank == 0:
            raise ValueError('Feature (key: {}) cannot have rank 0. Given: {}'.format(key, feature_tensor))
        return feature_tensor if rank != 1 else expand_dims(feature_tensor)
    with ops.control_dependencies([check_ops.assert_positive(array_ops.rank(feature_tensor), message='Feature (key: {}) cannot have rank 0. Given: {}'.format(key, feature_tensor))]):
        return cond.cond(math_ops.equal(1, array_ops.rank(feature_tensor)), lambda: expand_dims(feature_tensor), lambda: feature_tensor)