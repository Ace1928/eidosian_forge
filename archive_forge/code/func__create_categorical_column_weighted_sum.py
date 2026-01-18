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
def _create_categorical_column_weighted_sum(column, builder, units, sparse_combiner, weight_collections, trainable, weight_var=None):
    """Create a weighted sum of a categorical column for linear_model.

  Note to maintainer: As implementation details, the weighted sum is
  implemented via embedding_lookup_sparse toward efficiency. Mathematically,
  they are the same.

  To be specific, conceptually, categorical column can be treated as multi-hot
  vector. Say:

  ```python
    x = [0 0 1]  # categorical column input
    w = [a b c]  # weights
  ```
  The weighted sum is `c` in this case, which is same as `w[2]`.

  Another example is

  ```python
    x = [0 1 1]  # categorical column input
    w = [a b c]  # weights
  ```
  The weighted sum is `b + c` in this case, which is same as `w[2] + w[3]`.

  For both cases, we can implement weighted sum via embedding_lookup with
  sparse_combiner = "sum".
  """
    sparse_tensors = column._get_sparse_tensors(builder, weight_collections=weight_collections, trainable=trainable)
    id_tensor = sparse_ops.sparse_reshape(sparse_tensors.id_tensor, [array_ops.shape(sparse_tensors.id_tensor)[0], -1])
    weight_tensor = sparse_tensors.weight_tensor
    if weight_tensor is not None:
        weight_tensor = sparse_ops.sparse_reshape(weight_tensor, [array_ops.shape(weight_tensor)[0], -1])
    if weight_var is not None:
        weight = weight_var
    else:
        weight = variable_scope.get_variable(name='weights', shape=(column._num_buckets, units), initializer=init_ops.zeros_initializer(), trainable=trainable, collections=weight_collections)
    return embedding_ops.safe_embedding_lookup_sparse(weight, id_tensor, sparse_weights=weight_tensor, combiner=sparse_combiner, name='weighted_sum')