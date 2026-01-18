import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _matrix_exp_pade5(matrix):
    """5th-order Pade approximant for matrix exponential."""
    b = [30240.0, 15120.0, 3360.0, 420.0, 30.0]
    b = [constant_op.constant(x, matrix.dtype) for x in b]
    ident = linalg_ops.eye(array_ops.shape(matrix)[-2], batch_shape=array_ops.shape(matrix)[:-2], dtype=matrix.dtype)
    matrix_2 = math_ops.matmul(matrix, matrix)
    matrix_4 = math_ops.matmul(matrix_2, matrix_2)
    tmp = matrix_4 + b[3] * matrix_2 + b[1] * ident
    matrix_u = math_ops.matmul(matrix, tmp)
    matrix_v = b[4] * matrix_4 + b[2] * matrix_2 + b[0] * ident
    return (matrix_u, matrix_v)