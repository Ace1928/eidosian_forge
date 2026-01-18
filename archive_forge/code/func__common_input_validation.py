import numpy as np
import operator
from . import (linear_sum_assignment, OptimizeResult)
from ._optimize import _check_unknown_options
from scipy._lib._util import check_random_state
import itertools
def _common_input_validation(A, B, partial_match):
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    if partial_match is None:
        partial_match = np.array([[], []]).T
    partial_match = np.atleast_2d(partial_match).astype(int)
    msg = None
    if A.shape[0] != A.shape[1]:
        msg = '`A` must be square'
    elif B.shape[0] != B.shape[1]:
        msg = '`B` must be square'
    elif A.ndim != 2 or B.ndim != 2:
        msg = '`A` and `B` must have exactly two dimensions'
    elif A.shape != B.shape:
        msg = '`A` and `B` matrices must be of equal size'
    elif partial_match.shape[0] > A.shape[0]:
        msg = '`partial_match` can have only as many seeds as there are nodes'
    elif partial_match.shape[1] != 2:
        msg = '`partial_match` must have two columns'
    elif partial_match.ndim != 2:
        msg = '`partial_match` must have exactly two dimensions'
    elif (partial_match < 0).any():
        msg = '`partial_match` must contain only positive indices'
    elif (partial_match >= len(A)).any():
        msg = '`partial_match` entries must be less than number of nodes'
    elif not len(set(partial_match[:, 0])) == len(partial_match[:, 0]) or not len(set(partial_match[:, 1])) == len(partial_match[:, 1]):
        msg = '`partial_match` column entries must be unique'
    if msg is not None:
        raise ValueError(msg)
    return (A, B, partial_match)