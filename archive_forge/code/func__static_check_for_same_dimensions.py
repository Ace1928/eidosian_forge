import abc
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_full_matrix
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_lower_triangular
def _static_check_for_same_dimensions(operators):
    """ValueError if operators determined to have different dimensions."""
    if len(operators) < 2:
        return
    domain_dimensions = [(op.name, tensor_shape.dimension_value(op.domain_dimension)) for op in operators if tensor_shape.dimension_value(op.domain_dimension) is not None]
    if len(set((value for name, value in domain_dimensions))) > 1:
        raise ValueError(f'All `operators` must have the same `domain_dimension`. Received: {domain_dimensions}.')
    range_dimensions = [(op.name, tensor_shape.dimension_value(op.range_dimension)) for op in operators if tensor_shape.dimension_value(op.range_dimension) is not None]
    if len(set((value for name, value in range_dimensions))) > 1:
        raise ValueError(f'All operators must have the same `range_dimension`. Received: {range_dimensions}.')