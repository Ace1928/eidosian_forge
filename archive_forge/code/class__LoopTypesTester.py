import functools
import itertools
import sys
import warnings
import threading
import operator
import numpy as np
import unittest
from numba import guvectorize, njit, typeof, vectorize
from numba.core import types
from numba.np.numpy_support import from_dtype
from numba.core.errors import LoweringError, TypingError
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.typing.npydecl import supported_ufuncs
from numba.np import numpy_support
from numba.core.registry import cpu_target
from numba.core.base import BaseContext
from numba.np import ufunc_db
class _LoopTypesTester(TestCase):
    """Test code generation for the different loop types defined by ufunc.

    This test relies on class variables to configure the test. Subclasses
    of this class can just override some of these variables to check other
    ufuncs in a different compilation context. The variables supported are:

    _funcs: the ufuncs to test
    _skip_types: letter types that force skipping the loop when testing
                 if present in the NumPy ufunc signature.
    _supported_types: only test loops where all the types in the loop
                      signature are in this collection. If unset, all.

    Note that both, _skip_types and _supported_types must be met for a loop
    to be tested.

    The NumPy ufunc signature has a form like 'ff->f' (for a binary ufunc
    loop taking 2 floats and resulting in a float). In a NumPy ufunc object
    you can get a list of supported signatures by accessing the attribute
    'types'.
    """
    _skip_types = 'OegG'
    _ulps = {('arccos', 'F'): 2, ('arcsin', 'D'): 4, ('arcsin', 'F'): 4, ('log10', 'D'): 5, ('tanh', 'F'): 2, ('cbrt', 'd'): 2, ('logaddexp2', 'd'): 2}

    def _arg_for_type(self, a_letter_type, index=0):
        """return a suitable array argument for testing the letter type"""
        if a_letter_type in 'bhilq':
            return np.array([1, 4, 0, -2], dtype=a_letter_type)
        if a_letter_type in 'BHILQ':
            return np.array([1, 2, 4, 0], dtype=a_letter_type)
        elif a_letter_type in '?':
            return np.array([True, False, False, True], dtype=a_letter_type)
        elif a_letter_type[0] == 'm':
            if len(a_letter_type) == 1:
                a_letter_type = 'm8[D]'
            return np.array([2, -3, 'NaT', 0], dtype=a_letter_type)
        elif a_letter_type[0] == 'M':
            if len(a_letter_type) == 1:
                a_letter_type = 'M8[D]'
            return np.array(['Nat', 1, 25, 0], dtype=a_letter_type)
        elif a_letter_type in 'fd':
            return np.array([1.5, -3.5, 0.0, float('nan')], dtype=a_letter_type)
        elif a_letter_type in 'FD':
            if sys.platform != 'win32':
                negzero = -(0.0 + 1j)
            else:
                negzero = 0.0 - 1j
            return np.array([negzero, 1.5 + 1.5j, 1j * float('nan'), 0j], dtype=a_letter_type)
        else:
            raise RuntimeError('type %r not understood' % (a_letter_type,))

    def _check_loop(self, fn, ufunc, loop):
        letter_types = loop[:ufunc.nin] + loop[-ufunc.nout:]
        supported_types = getattr(self, '_supported_types', [])
        if supported_types and any((l not in supported_types for l in letter_types)):
            return
        skip_types = getattr(self, '_skip_types', [])
        if any((l in skip_types for l in letter_types)):
            return
        required_types = getattr(self, '_required_types', [])
        if required_types and (not any((l in letter_types for l in required_types))):
            return
        self._check_ufunc_with_dtypes(fn, ufunc, letter_types)

    def _check_ufunc_with_dtypes(self, fn, ufunc, dtypes):
        dtypes_with_units = []
        for t in dtypes:
            if t in ('m', 'M'):
                t = t + '8[D]'
            dtypes_with_units.append(t)
        arg_dty = [np.dtype(t) for t in dtypes_with_units]
        arg_nbty = tuple([types.Array(from_dtype(t), 1, 'C') for t in arg_dty])
        cfunc = njit(arg_nbty)(fn)
        c_args = [self._arg_for_type(t, index=index).repeat(2) for index, t in enumerate(dtypes)]
        for arr in c_args:
            self.random.shuffle(arr)
        py_args = [a.copy() for a in c_args]
        cfunc(*c_args)
        fn(*py_args)
        for dtype, py_arg, c_arg in zip(arg_dty, py_args, c_args):
            py_arg, c_arg = self._fixup_results(dtype, py_arg, c_arg)
            typechar = c_arg.dtype.char
            ulps = self._ulps.get((ufunc.__name__, typechar), 1)
            prec = 'single' if typechar in 'fF' else 'exact'
            prec = 'double' if typechar in 'dD' else prec
            msg = '\n'.join(["ufunc '{0}' arrays differ ({1}):", 'args: {2}', 'expected {3}', 'got {4}'])
            msg = msg.format(ufunc.__name__, c_args, prec, py_arg, c_arg)
            self.assertPreciseEqual(py_arg, c_arg, prec=prec, msg=msg, ulps=ulps)

    def _fixup_results(self, dtype, py_arg, c_arg):
        return (py_arg, c_arg)

    @classmethod
    def _check_ufunc_loops(cls, ufunc):
        for loop in ufunc.types:
            cls._inject_test(ufunc, loop)

    @classmethod
    def _inject_test(cls, ufunc, loop):

        def test_template(self):
            fn = _make_ufunc_usecase(ufunc)
            self._check_loop(fn, ufunc, loop)
        setattr(cls, 'test_{0}_{1}'.format(ufunc.__name__, loop.replace('->', '_')), test_template)

    @classmethod
    def autogenerate(cls):
        for ufunc in cls._ufuncs:
            cls._check_ufunc_loops(ufunc)