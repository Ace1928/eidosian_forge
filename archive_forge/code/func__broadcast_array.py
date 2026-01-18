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
def _broadcast_array(self, ary, newdim, innerdim):
    newshape = (newdim,) + innerdim
    if ary.shape == newshape:
        return ary
    elif len(ary.shape) < len(newshape):
        assert newshape[-len(ary.shape):] == ary.shape, 'cannot add dim and reshape at the same time'
        return self._broadcast_add_axis(ary, newshape)
    else:
        return ary.reshape(*newshape)