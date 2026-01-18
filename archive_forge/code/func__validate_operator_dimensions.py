from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _validate_operator_dimensions(self):
    """Check that `operators` have compatible dimensions."""
    for i in range(1, len(self.operators)):
        for j in range(i):
            op = self.operators[i][j]
            above_op = self.operators[i - 1][j]
            right_op = self.operators[i][j + 1]
            if op.domain_dimension is not None and above_op.domain_dimension is not None:
                if op.domain_dimension != above_op.domain_dimension:
                    raise ValueError(f'Argument `operators[{i}][{j}].domain_dimension` ({op.domain_dimension}) must be the same as `operators[{i - 1}][{j}].domain_dimension` ({above_op.domain_dimension}).')
            if op.range_dimension is not None and right_op.range_dimension is not None:
                if op.range_dimension != right_op.range_dimension:
                    raise ValueError(f'Argument `operators[{i}][{j}].range_dimension` ({op.range_dimension}) must be the same as `operators[{i}][{j + 1}].range_dimension` ({right_op.range_dimension}).')