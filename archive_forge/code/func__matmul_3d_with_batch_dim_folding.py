import functools
import typing
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _matmul_3d_with_batch_dim_folding(a, b, **kwargs):
    """Multiply batches of 2D matrices where only `a.shape[1]` is ragged.

  Args:
    a: A RaggedTensor with `shape=[B, (I), J]`.  (ragged_rank must be 1.)
    b: A Tensor with `shape=[B, J, K]`
    **kwargs: Additional arguments for `tf.matmul` (e.g. transpose_a).
      transpose_a and adjoint_a must not be true.

  Returns:
    A RaggedTensor with `shape=[B, (I), K].
  """
    reshaped_a = array_ops.expand_dims(a.values, 1)
    reshaped_b = array_ops.repeat(b, a.row_lengths(), axis=0)
    flat_result = math_ops.matmul(reshaped_a, reshaped_b, **kwargs)
    return a.with_values(array_ops.squeeze(flat_result, axis=1))