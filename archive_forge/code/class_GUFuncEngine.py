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
class GUFuncEngine(object):
    """Determine how to broadcast and execute a gufunc
    base on input shape and signature
    """

    @classmethod
    def from_signature(cls, signature):
        return cls(*parse_signature(signature))

    def __init__(self, inputsig, outputsig):
        self.sin = inputsig
        self.sout = outputsig
        self.nin = len(self.sin)
        self.nout = len(self.sout)

    def schedule(self, ishapes):
        if len(ishapes) != self.nin:
            raise TypeError('invalid number of input argument')
        symbolmap = {}
        outer_shapes = []
        inner_shapes = []
        for argn, (shape, symbols) in enumerate(zip(ishapes, self.sin)):
            argn += 1
            inner_ndim = len(symbols)
            if len(shape) < inner_ndim:
                fmt = 'arg #%d: insufficient inner dimension'
                raise ValueError(fmt % (argn,))
            if inner_ndim:
                inner_shape = shape[-inner_ndim:]
                outer_shape = shape[:-inner_ndim]
            else:
                inner_shape = ()
                outer_shape = shape
            for axis, (dim, sym) in enumerate(zip(inner_shape, symbols)):
                axis += len(outer_shape)
                if sym in symbolmap:
                    if symbolmap[sym] != dim:
                        fmt = 'arg #%d: shape[%d] mismatch argument'
                        raise ValueError(fmt % (argn, axis))
                symbolmap[sym] = dim
            outer_shapes.append(outer_shape)
            inner_shapes.append(inner_shape)
        oshapes = []
        for outsig in self.sout:
            oshape = []
            for sym in outsig:
                oshape.append(symbolmap[sym])
            oshapes.append(tuple(oshape))
        sizes = [reduce(operator.mul, s, 1) for s in outer_shapes]
        largest_i = np.argmax(sizes)
        loopdims = outer_shapes[largest_i]
        pinned = [False] * self.nin
        for i, d in enumerate(outer_shapes):
            if d != loopdims:
                if d == (1,) or d == ():
                    pinned[i] = True
                else:
                    fmt = 'arg #%d: outer dimension mismatch'
                    raise ValueError(fmt % (i + 1,))
        return GUFuncSchedule(self, inner_shapes, oshapes, loopdims, pinned)