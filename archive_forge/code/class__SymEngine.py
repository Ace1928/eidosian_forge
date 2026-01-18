from __future__ import (absolute_import, division, print_function)
import os
import sys
import numpy as np
from .util import banded_jacobian, sparse_jacobian_csc, sparse_jacobian_csr
class _SymEngine(_Base):
    _dummy_counter = [0]

    def __init__(self):
        self.__sym_backend__ = __import__('symengine')
        self._cse = getattr(self.__sym_backend__, 'cse', None)

    def Matrix(self, *args, **kwargs):
        return self.DenseMatrix(*args, **kwargs)

    def real_symarray(self, prefix, shape):
        return self.symarray(prefix, shape)

    def Dummy(self):
        self._dummy_counter[0] += 1
        return self.Symbol('Dummy_' + str(self._dummy_counter[0] - 1))

    def numbered_symbols(self, prefix='x', cls=None, start=0, exclude=None, *args):
        exclude = set(exclude or [])
        if cls is None:
            cls = self.Symbol
        while True:
            name = '%s%s' % (prefix, start)
            s = cls(name, *args)
            if s not in exclude:
                yield s
            start += 1

    def cse(self, exprs, symbols=None):
        if self._cse is None:
            raise NotImplementedError('CSE not yet supported in symengine version %s' % self.__sym_backend__.__version__)
        if symbols is None:
            symbols = self.numbered_symbols()
        else:
            symbols = iter(symbols)
        old_repl, old_reduced = self._cse(exprs)
        if not old_repl:
            return (old_repl, old_reduced)
        old_cse_symbols, old_cses = zip(*old_repl)
        cse_symbols = [next(symbols) for _ in range(len(old_cse_symbols))]
        subsd = dict(zip(old_cse_symbols, cse_symbols))
        cses = [c.xreplace(subsd) for c in old_cses]
        reduced = [e.xreplace(subsd) for e in old_reduced]
        return (list(zip(cse_symbols, cses)), reduced)