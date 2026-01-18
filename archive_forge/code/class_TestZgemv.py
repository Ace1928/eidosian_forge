from numpy import float32, float64, complex64, complex128, arange, array, \
from scipy.linalg import _fblas as fblas
from numpy.testing import assert_array_equal, \
import pytest
class TestZgemv(BaseGemv):
    blas_func = fblas.zgemv
    dtype = complex128