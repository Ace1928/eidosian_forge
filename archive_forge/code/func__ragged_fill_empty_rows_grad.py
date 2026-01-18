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
@ops.RegisterGradient('RaggedFillEmptyRows')
def _ragged_fill_empty_rows_grad(op, unused_grad_output_indices, output_grad_values, unused_grad_empty_row_indicator, unused_grad_reverse_index_map):
    """Gradients for RaggedFillEmptyRows."""
    reverse_index_map = op.outputs[3]
    d_values, d_default_value = gen_ragged_array_ops.ragged_fill_empty_rows_grad(reverse_index_map=reverse_index_map, grad_values=output_grad_values)
    return [None, d_values, None, d_default_value]