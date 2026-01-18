from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import operator
import warnings
from functools import reduce
import numpy as np
from numba.np.ufunc.ufuncbuilder import _BaseUFuncBuilder, parse_identity
from numba.core import types, sigutils
from numba.core.typing import signature
from numba.np.ufunc.sigparse import parse_signature
def _pairwise_broadcast(shape1, shape2):
    """
    Raises
    ------
    ValueError if broadcast fails
    """
    shape1, shape2 = map(tuple, [shape1, shape2])
    while len(shape1) < len(shape2):
        shape1 = (1,) + shape1
    while len(shape1) > len(shape2):
        shape2 = (1,) + shape2
    return tuple((_broadcast_axis(a, b) for a, b in zip(shape1, shape2)))