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
def _verify_static_batch_size_equality(tensors, columns):
    """Validates that the first dim (batch size) of all tensors are equal or None.

  Args:
    tensors: list of tensors to check.
    columns: list of feature columns matching tensors. Will be used for error
      messaging.

  Raises:
    ValueError: if one of the tensors has a variant batch size
  """
    expected_batch_size = None
    for i in range(0, len(tensors)):
        if tensors[i].shape.dims[0].value is not None:
            if expected_batch_size is None:
                bath_size_column_index = i
                expected_batch_size = tensors[i].shape.dims[0]
            elif not expected_batch_size.is_compatible_with(tensors[i].shape.dims[0]):
                raise ValueError('Batch size (first dimension) of each feature must be same. Batch size of columns ({}, {}): ({}, {})'.format(columns[bath_size_column_index].name, columns[i].name, expected_batch_size, tensors[i].shape.dims[0]))