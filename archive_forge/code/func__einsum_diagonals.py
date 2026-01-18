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
def _einsum_diagonals(input_subscripts, operands):
    """Compute diagonal for each operand

    This function mutates args.
    """
    for idx in range(len(input_subscripts)):
        sub = input_subscripts[idx]
        arr = operands[idx]
        if len(set(sub)) < len(sub):
            axeses = {}
            for axis, label in enumerate(sub):
                axeses.setdefault(label, []).append(axis)
            axeses = list(axeses.items())
            for label, axes in axeses:
                if options['broadcast_diagonal']:
                    axes = [axis for axis in axes if arr.shape[axis] != 1]
                dims = {arr.shape[axis] for axis in axes}
                if len(dims) >= 2:
                    dim0 = dims.pop()
                    dim1 = dims.pop()
                    raise ValueError("dimensions in operand %d for collapsing index '%s' don't match (%d != %d)" % (idx, _chr(label), dim0, dim1))
            sub, axeses = zip(*axeses)
            input_subscripts[idx] = list(sub)
            operands[idx] = _transpose_ex(arr, axeses)