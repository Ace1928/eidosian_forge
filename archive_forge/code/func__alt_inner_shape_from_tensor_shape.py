import abc
from typing import Any, Iterable, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.ragged.row_partition import RowPartitionSpec
from tensorflow.python.types import core
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _alt_inner_shape_from_tensor_shape(shape, dtype, new_inner_rank):
    """Helper for _alt_inner_shape, used directly in _with_num_row_partitions."""
    if new_inner_rank == 1:
        return constant_op.constant([shape.num_elements()], dtype=dtype)
    new_inner_rank_tail_length = new_inner_rank - 1
    inner_shape_tail = shape[-new_inner_rank_tail_length:].as_list()
    first_dim = shape[:-new_inner_rank_tail_length].num_elements()
    return constant_op.constant([first_dim] + inner_shape_tail, dtype=dtype)