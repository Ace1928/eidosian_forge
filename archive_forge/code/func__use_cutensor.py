import copy
import itertools
import operator
import string
import warnings
import cupy
from cupy._core import _accelerator
from cupy import _util
from cupy.linalg._einsum_opt import _greedy_path
from cupy.linalg._einsum_opt import _optimal_path
from cupy.linalg._einsum_cutn import _try_use_cutensornet
def _use_cutensor(dtype0, sub0, dtype1, sub1, batch_dims, contract_dims):
    if not cutensor.check_availability('contraction'):
        return False
    if dtype0 != dtype1:
        return False
    if dtype0 not in (cupy.float32, cupy.float64, cupy.complex64, cupy.complex128):
        return False
    return True