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
class TestTimedeltaArithmeticNoPython(TestTimedeltaArithmetic):
    jitargs = dict(nopython=True)

    def test_int_cast(self):
        f = self.jit(int_cast_usecase)

        def check(a):
            self.assertPreciseEqual(f(a), int(a))
        for delta, unit in ((3, 'ns'), (-4, 'ns'), (30000, 'ns'), (-40000000, 'ns'), (1, 'Y')):
            check(TD(delta, unit).astype('timedelta64[ns]'))
        for time in ('2014', '2016', '2000', '2014-02', '2014-03', '2014-04', '2016-02', '2000-12-31', '2014-01-16', '2014-01-05', '2014-01-07', '2014-01-06', '2014-02-02', '2014-02-27', '2014-02-16', '2014-03-01', '2000-01-01T01:02:03.002Z', '2000-01-01T01:02:03Z'):
            check(DT(time).astype('datetime64[ns]'))
        with self.assertRaises(TypingError, msg='Only datetime64[ns] can be ' + 'converted, but got ' + 'datetime64[y]'):
            f(DT('2014'))