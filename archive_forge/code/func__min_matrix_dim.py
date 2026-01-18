import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export
def _min_matrix_dim(self):
    """Minimum of domain/range dimension, if statically available, else None."""
    domain_dim = self.domain_dimension.value
    range_dim = self.range_dimension.value
    if domain_dim is None or range_dim is None:
        return None
    return min(domain_dim, range_dim)