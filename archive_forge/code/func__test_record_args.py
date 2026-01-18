import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def _test_record_args(self, revargs):
    npval = self.refsample1d.copy()[0]
    nbval = self.nbsample1d.copy()[0]
    attrs = 'abc'
    valtypes = (types.float64, types.int16, types.complex64)
    values = (1.23, 12345, 123 + 456j)
    with self.assertRefCount(nbval):
        for attr, valtyp, val in zip(attrs, valtypes, values):
            expected = getattr(npval, attr)
            nbrecord = numpy_support.from_dtype(recordtype)
            if revargs:
                prefix = 'get_record_rev_'
                argtypes = (valtyp, nbrecord)
                args = (val, nbval)
            else:
                prefix = 'get_record_'
                argtypes = (nbrecord, valtyp)
                args = (nbval, val)
            pyfunc = globals()[prefix + attr]
            cfunc = self.get_cfunc(pyfunc, argtypes)
            got = cfunc(*args)
            try:
                self.assertEqual(expected, got)
            except AssertionError:
                import llvmlite.binding as ll
                if attr != 'c':
                    raise
                triple = 'armv7l-unknown-linux-gnueabihf'
                if ll.get_default_triple() != triple:
                    raise
                self.assertEqual(val, got)
            else:
                self.assertEqual(nbval[attr], val)
            del got, expected, args