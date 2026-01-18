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
def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    """Returns dense `Tensor` representing feature.

    Args:
      inputs: A `_LazyBuilder` object to access inputs.
      weight_collections: Unused `weight_collections` since no variables are
        created in this function.
      trainable: Unused `trainable` bool since no variables are created in this
        function.

    Returns:
      Dense `Tensor` created within `_transform_feature`.

    Raises:
      ValueError: If `categorical_column` is a `_SequenceCategoricalColumn`.
    """
    del weight_collections
    del trainable
    if isinstance(self.categorical_column, _SequenceCategoricalColumn):
        raise ValueError('In indicator_column: {}. categorical_column must not be of type _SequenceCategoricalColumn. Suggested fix A: If you wish to use input_layer, use a non-sequence categorical_column_with_*. Suggested fix B: If you wish to create sequence input, use sequence_input_layer instead of input_layer. Given (type {}): {}'.format(self.name, type(self.categorical_column), self.categorical_column))
    return inputs.get(self)