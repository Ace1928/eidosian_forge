import typing
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _concat_ragged_splits(splits_list):
    """Concatenates a list of RaggedTensor splits to form a single splits."""
    pieces = [splits_list[0]]
    splits_offset = splits_list[0][-1]
    for splits in splits_list[1:]:
        pieces.append(splits[1:] + splits_offset)
        splits_offset += splits[-1]
    return array_ops.concat(pieces, axis=0)