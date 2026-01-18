import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
class TestNPMachineParameters(TestCase):
    template = '\ndef foo():\n    ty = np.%s\n    return np.%s(ty)\n'

    def check(self, func, attrs, *args):
        pyfunc = func
        cfunc = jit(nopython=True)(pyfunc)
        expected = pyfunc(*args)
        got = cfunc(*args)
        for attr in attrs:
            self.assertPreciseEqual(getattr(expected, attr), getattr(got, attr))

    def create_harcoded_variant(self, basefunc, ty):
        tystr = ty.__name__
        basestr = basefunc.__name__
        funcstr = self.template % (tystr, basestr)
        eval(compile(funcstr, '<string>', 'exec'))
        return locals()['foo']

    @unittest.skipIf(numpy_version >= (1, 24), 'NumPy < 1.24 required')
    def test_MachAr(self):
        attrs = ('ibeta', 'it', 'machep', 'eps', 'negep', 'epsneg', 'iexp', 'minexp', 'xmin', 'maxexp', 'xmax', 'irnd', 'ngrd', 'epsilon', 'tiny', 'huge', 'precision', 'resolution')
        self.check(machar, attrs)

    def test_finfo(self):
        types = [np.float32, np.float64, np.complex64, np.complex128]
        attrs = ('eps', 'epsneg', 'iexp', 'machep', 'max', 'maxexp', 'negep', 'nexp', 'nmant', 'precision', 'resolution', 'tiny', 'bits')
        for ty in types:
            self.check(finfo, attrs, ty(1))
            hc_func = self.create_harcoded_variant(np.finfo, ty)
            self.check(hc_func, attrs)
        with self.assertRaises(TypingError) as raises:
            cfunc = jit(nopython=True)(finfo_machar)
            cfunc(7.0)
        msg = "Unknown attribute 'machar' of type finfo"
        self.assertIn(msg, str(raises.exception))
        with self.assertTypingError():
            cfunc = jit(nopython=True)(finfo)
            cfunc(np.int32(7))

    def test_iinfo(self):
        types = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
        attrs = ('min', 'max', 'bits')
        for ty in types:
            self.check(iinfo, attrs, ty(1))
            hc_func = self.create_harcoded_variant(np.iinfo, ty)
            self.check(hc_func, attrs)
        with self.assertTypingError():
            cfunc = jit(nopython=True)(iinfo)
            cfunc(np.float64(7))

    @unittest.skipUnless(numpy_version < (1, 24), 'Needs NumPy < 1.24')
    @TestCase.run_test_in_subprocess
    def test_np_MachAr_deprecation_np122(self):
        msg = '.*`np.MachAr` is deprecated \\(NumPy 1.22\\)'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('ignore')
            warnings.filterwarnings('always', message=msg, category=NumbaDeprecationWarning)
            f = njit(lambda: np.MachAr().eps)
            f()
        self.assertEqual(len(w), 1)
        self.assertIn('`np.MachAr` is deprecated', str(w[0]))