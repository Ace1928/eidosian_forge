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
def _parse_ellipsis_subscript(subscript, idx, ndim=None, ellipsis_len=None):
    """Parse a subscript that may contain ellipsis

    Args:
        subscript (str): An einsum subscript of an operand or an output. '...'
            should be replaced by '@'.
        idx (int or None): For error messages, give int idx for the idx-th
            operand or None for the output.
        ndim (int, optional): ndim of the operand
        ellipsis_len (int, optional): number of broadcast dimensions of the
            output.

    Returns:
        list of ints: The parsed subscript

    """
    subs = subscript.split('@')
    if len(subs) == 1:
        sub, = subs
        if ndim is not None and len(sub) != ndim:
            if len(sub) > ndim:
                raise ValueError('einstein sum subscripts string %s contains too many subscripts for operand %d' % (sub, idx))
            raise ValueError("operand %d has more dimensions than subscripts string %s given in einstein sum, but no '...' ellipsis provided to broadcast the extra dimensions." % (idx, sub))
        return [ord(label) for label in sub]
    elif len(subs) == 2:
        left_sub, right_sub = subs
        if ndim is not None:
            ellipsis_len = ndim - (len(left_sub) + len(right_sub))
        if ellipsis_len < 0:
            raise ValueError('einstein sum subscripts string %s...%s contains too many subscripts for operand %d' % (left_sub, right_sub, idx))
        ret = []
        ret.extend((ord(label) for label in left_sub))
        ret.extend(range(-ellipsis_len, 0))
        ret.extend((ord(label) for label in right_sub))
        return ret
    else:
        raise ValueError("einstein sum subscripts string contains a '.' that is not part of an ellipsis ('...') " + ('in the output' if idx is None else 'for operand %d' % idx))