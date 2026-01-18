import contextlib
import itertools
import re
import unittest
import warnings
import numpy as np
from numba import jit, vectorize, njit
from numba.np.numpy_support import numpy_version
from numba.core import types, config
from numba.core.errors import TypingError
from numba.tests.support import TestCase, tag, skip_parfors_unsupported
from numba.np import npdatetime_helpers, numpy_support
class TestMiscCompiling(TestCase):

    def test_jit_explicit_signature(self):

        def _check_explicit_signature(sig):
            f = jit(sig, nopython=True)(add_usecase)
            args = (DT(1, 'ms'), TD(2, 'us'))
            expected = add_usecase(*args)
            self.assertPreciseEqual(f(*args), expected)
        sig = types.NPDatetime('us')(types.NPDatetime('ms'), types.NPTimedelta('us'))
        _check_explicit_signature(sig)
        sig = "NPDatetime('us')(NPDatetime('ms'), NPTimedelta('us'))"
        _check_explicit_signature(sig)

    def test_vectorize_explicit_signature(self):

        def _check_explicit_signature(sig):
            f = vectorize([sig], nopython=True)(mul_usecase)
            self.assertPreciseEqual(f(TD(2), 3), TD(6))
        sig = types.NPTimedelta('s')(types.NPTimedelta('s'), types.int64)
        _check_explicit_signature(sig)
        sig = "NPTimedelta('s')(NPTimedelta('s'), int64)"
        _check_explicit_signature(sig)

    def test_constant_datetime(self):

        def check(const):
            pyfunc = make_add_constant(const)
            f = jit(nopython=True)(pyfunc)
            x = TD(4, 'D')
            expected = pyfunc(x)
            self.assertPreciseEqual(f(x), expected)
        check(DT('2001-01-01'))
        check(DT('NaT', 'D'))

    def test_constant_timedelta(self):

        def check(const):
            pyfunc = make_add_constant(const)
            f = jit(nopython=True)(pyfunc)
            x = TD(4, 'D')
            expected = pyfunc(x)
            self.assertPreciseEqual(f(x), expected)
        check(TD(4, 'D'))
        check(TD(-4, 'D'))
        check(TD('NaT', 'D'))