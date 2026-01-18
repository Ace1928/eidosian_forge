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
def _tridiagonal_solve_compact_format(diagonals, rhs, transpose_rhs, conjugate_rhs, partial_pivoting, perturb_singular, name):
    """Helper function used after the input has been cast to compact form."""
    diags_rank, rhs_rank = (diagonals.shape.rank, rhs.shape.rank)
    if diags_rank:
        if diags_rank < 2:
            raise ValueError('Expected diagonals to have rank at least 2, got {}'.format(diags_rank))
        if rhs_rank and rhs_rank != diags_rank and (rhs_rank != diags_rank - 1):
            raise ValueError('Expected the rank of rhs to be {} or {}, got {}'.format(diags_rank - 1, diags_rank, rhs_rank))
        if rhs_rank and (not diagonals.shape[:-2].is_compatible_with(rhs.shape[:diags_rank - 2])):
            raise ValueError('Batch shapes {} and {} are incompatible'.format(diagonals.shape[:-2], rhs.shape[:diags_rank - 2]))
    if diagonals.shape[-2] and diagonals.shape[-2] != 3:
        raise ValueError('Expected 3 diagonals got {}'.format(diagonals.shape[-2]))

    def check_num_lhs_matches_num_rhs():
        if diagonals.shape[-1] and rhs.shape[-2] and (diagonals.shape[-1] != rhs.shape[-2]):
            raise ValueError('Expected number of left-hand sided and right-hand sides to be equal, got {} and {}'.format(diagonals.shape[-1], rhs.shape[-2]))
    if rhs_rank and diags_rank and (rhs_rank == diags_rank - 1):
        if conjugate_rhs:
            rhs = math_ops.conj(rhs)
        rhs = array_ops.expand_dims(rhs, -1)
        check_num_lhs_matches_num_rhs()
        return array_ops.squeeze(linalg_ops.tridiagonal_solve(diagonals, rhs, partial_pivoting, perturb_singular, name), -1)
    if transpose_rhs:
        rhs = array_ops.matrix_transpose(rhs, conjugate=conjugate_rhs)
    elif conjugate_rhs:
        rhs = math_ops.conj(rhs)
    check_num_lhs_matches_num_rhs()
    return linalg_ops.tridiagonal_solve(diagonals, rhs, partial_pivoting, perturb_singular, name)