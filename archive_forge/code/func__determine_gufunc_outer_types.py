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
def _determine_gufunc_outer_types(argtys, dims):
    for at, nd in zip(argtys, dims):
        if isinstance(at, types.Array):
            yield at.copy(ndim=nd + 1)
        else:
            if nd > 0:
                raise ValueError('gufunc signature mismatch: ndim>0 for scalar')
            yield types.Array(dtype=at, ndim=1, layout='A')