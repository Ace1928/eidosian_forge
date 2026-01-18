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
def _matrix_exp_pade13(matrix):
    """13th-order Pade approximant for matrix exponential."""
    b = [6.476475253248e+16, 3.238237626624e+16, 7771770303897600.0, 1187353796428800.0, 129060195264000.0, 10559470521600.0, 670442572800.0, 33522128640.0, 1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0]
    b = [constant_op.constant(x, matrix.dtype) for x in b]
    ident = linalg_ops.eye(array_ops.shape(matrix)[-2], batch_shape=array_ops.shape(matrix)[:-2], dtype=matrix.dtype)
    matrix_2 = math_ops.matmul(matrix, matrix)
    matrix_4 = math_ops.matmul(matrix_2, matrix_2)
    matrix_6 = math_ops.matmul(matrix_4, matrix_2)
    tmp_u = math_ops.matmul(matrix_6, matrix_6 + b[11] * matrix_4 + b[9] * matrix_2) + b[7] * matrix_6 + b[5] * matrix_4 + b[3] * matrix_2 + b[1] * ident
    matrix_u = math_ops.matmul(matrix, tmp_u)
    tmp_v = b[12] * matrix_6 + b[10] * matrix_4 + b[8] * matrix_2
    matrix_v = math_ops.matmul(matrix_6, tmp_v) + b[6] * matrix_6 + b[4] * matrix_4 + b[2] * matrix_2 + b[0] * ident
    return (matrix_u, matrix_v)