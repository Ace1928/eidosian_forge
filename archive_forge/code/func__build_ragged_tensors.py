import collections
import re
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.tf_export import tf_export
def _build_ragged_tensors(serialized_shape, ragged_values, ragged_row_splits, ragged_inner_splits=None):
    """Builds RaggedTensors from the outputs of a parse op."""
    if ragged_inner_splits is not None:
        ragged_values = [ragged_tensor.RaggedTensor.from_row_splits(val, split, validate=False) for val, split in zip(ragged_values, ragged_inner_splits)]
    if serialized_shape.ndims == 0:
        return ragged_values
    else:
        return [ragged_tensor.RaggedTensor.from_row_splits(val, split, validate=False) for val, split in zip(ragged_values, ragged_row_splits)]