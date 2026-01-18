import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.gen_linalg_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _overdetermined(matrix, rhs, l2_regularizer):
    """Computes (A^H*A + l2_regularizer)^{-1} * A^H * rhs."""
    chol = _RegularizedGramianCholesky(matrix, l2_regularizer=l2_regularizer, first_kind=True)
    return cholesky_solve(chol, math_ops.matmul(matrix, rhs, adjoint_a=True))