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
def _get_out_shape(shape0, sub0, shape1, sub1, sub_out):
    extent = {}
    for size, i in zip(shape0 + shape1, sub0 + sub1):
        extent[i] = size
    out_shape = [extent[i] for i in sub_out]
    return out_shape