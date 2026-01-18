from typing import Any, Iterable, Optional, Text, Union, Dict
from absl import logging
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.tpu import tpu_embedding_base
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.types import core
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _ragged_embedding_lookup_with_reduce(table: tf_variables.Variable, ragged: ragged_tensor.RaggedTensor, weights: ragged_tensor.RaggedTensor, combiner: Text) -> core.Tensor:
    """Compute a ragged lookup followed by a reduce on axis 1.

  Args:
    table: The embedding table.
    ragged: A RaggedTensor of ids to look up.
    weights: A RaggedTensor of weights (or None).
    combiner: One of "mean", "sum", "sqrtn".

  Returns:
    A Tensor.
  """
    if weights is None:
        weights = array_ops.ones_like(ragged, dtype=table.dtype)
    weights = array_ops.expand_dims(weights, axis=2)
    ragged_result = embedding_ops.embedding_lookup(table, ragged)
    ragged_result = math_ops.reduce_sum(ragged_result * weights, axis=1)
    if combiner == 'mean':
        ragged_result = math_ops.div_no_nan(ragged_result, math_ops.reduce_sum(weights, axis=1))
    elif combiner == 'sqrtn':
        ragged_result = math_ops.div_no_nan(ragged_result, math_ops.sqrt(math_ops.reduce_sum(weights * weights, axis=1)))
    return ragged_result