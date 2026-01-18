from __future__ import division
from hypothesis import given, assume
from math import sqrt, floor
from blis.tests.common import *
from blis.py import gemm
def _stretch_matrix(data, m, n):
    orig_len = len(data)
    orig_m = m
    orig_n = n
    ratio = sqrt(len(data) / (m * n))
    m = int(floor(m * ratio))
    n = int(floor(n * ratio))
    data = np.ascontiguousarray(data[:m * n], dtype=data.dtype)
    return (data.reshape((m, n)), m, n)