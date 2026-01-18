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
def _make_transpose_axes(sub, b_dims, c_dims):
    bs = []
    cs = []
    ts = []
    for axis, label in enumerate(sub):
        if label in b_dims:
            bs.append((label, axis))
        elif label in c_dims:
            cs.append((label, axis))
        else:
            ts.append((label, axis))
    return (_tuple_sorted_by_0(bs), _tuple_sorted_by_0(cs), _tuple_sorted_by_0(ts))