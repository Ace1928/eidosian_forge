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
def _flatten_transpose(a, axeses):
    """Transpose and flatten each

    Args:
        a
        axeses (sequence of sequences of ints)

    Returns:
        aT: a with its axes permutated and flatten
        shapes: flattened shapes
    """
    transpose_axes = []
    shapes = []
    for axes in axeses:
        transpose_axes.extend(axes)
        shapes.append([a.shape[axis] for axis in axes])
    return (a.transpose(transpose_axes).reshape(tuple([cupy._core.internal.prod(shape) for shape in shapes])), shapes)