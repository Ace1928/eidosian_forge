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
def _multi_broadcast(*shapelist):
    """
    Raises
    ------
    ValueError if broadcast fails
    """
    assert shapelist
    result = shapelist[0]
    others = shapelist[1:]
    try:
        for i, each in enumerate(others, start=1):
            result = _pairwise_broadcast(result, each)
    except ValueError:
        raise ValueError('failed to broadcast argument #{0}'.format(i))
    else:
        return result