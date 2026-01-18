from __future__ import division
from hypothesis import given, assume
from math import sqrt, floor
from blis.tests.common import *
from blis.py import gemm
def _reshape_for_gemm(A, B, a_rows, a_cols, out_cols, dtype, trans_a=False, trans_b=False):
    A, a_rows, a_cols = _stretch_matrix(A, a_rows, a_cols)
    if len(B) < a_cols or a_cols < 1:
        return (None, None, None)
    b_cols = int(floor(len(B) / a_cols))
    B = np.ascontiguousarray(B.flatten()[:a_cols * b_cols], dtype=dtype)
    B = B.reshape((a_cols, b_cols))
    out_cols = B.shape[1]
    C = np.zeros(shape=(A.shape[0], B.shape[1]), dtype=dtype)
    if trans_a:
        A = np.ascontiguousarray(A.T, dtype=dtype)
    return (A, B, C)