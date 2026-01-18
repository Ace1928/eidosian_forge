from typing import Optional
from typing import Union
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_ragged_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _tile_ragged_splits(rt_input, multiples, const_multiples=None):
    """Builds nested_split tensors for a tiled `RaggedTensor`.

  Returns a list of split tensors that can be used to construct the
  `RaggedTensor` that tiles `rt_input` as specified by `multiples`.

  Args:
    rt_input: The `RaggedTensor` that is being tiled.
    multiples: A 1-D integer `tensor`, indicating how many times each dimension
      should be repeated.
    const_multiples: Optional constant value for multiples.  Used to skip tiling
      dimensions where `multiples=1`.

  Returns:
    A list of 1-D integer `Tensor`s (one for each ragged dimension in
    `rt_input`).

  #### Example:

  >>> rt = tf.ragged.constant([[1, 2], [3]])
  >>> _tile_ragged_splits(rt, [3, 2])
  [<tf.Tensor: shape=(7,), dtype=int64,
  numpy=array([ 0,  4,  6, 10, 12, 16, 18])>]
  """
    ragged_rank = rt_input.ragged_rank
    nested_splits = rt_input.nested_row_splits
    projected_splits = [{i: nested_splits[i]} for i in range(ragged_rank)]
    for src_axis in range(ragged_rank):
        for dst_axis in range(src_axis + 1, ragged_rank - 1):
            projected_splits[src_axis][dst_axis] = array_ops.gather(nested_splits[dst_axis], projected_splits[src_axis][dst_axis - 1])
    result_splits = []
    for axis in range(ragged_rank):
        input_lengths = nested_splits[axis][1:] - nested_splits[axis][:-1]
        output_lengths = input_lengths * multiples[axis + 1]
        repeats = 1
        for d in range(axis - 1, -1, -1):
            if const_multiples is None or const_multiples[d + 1] != 1:
                splits = projected_splits[d][axis - 1] * repeats
                output_lengths = ragged_util.repeat_ranges(output_lengths, splits, multiples[d + 1])
            repeats *= multiples[d + 1]
        output_lengths = array_ops.tile(output_lengths, multiples[:1])
        result_splits.append(ragged_util.lengths_to_splits(output_lengths))
    return result_splits