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
def _validate_num_operators(self):
    for i, row in enumerate(self.operators):
        if len(row) != i + 1:
            raise ValueError(f'Argument `operators[{i}]` must contain `{i + 1}` blocks. Received: {len(row)} blocks.')