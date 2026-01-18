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
class TestDatetimeArithmetic(TestCase):
    jitargs = dict(forceobj=True)

    def jit(self, pyfunc):
        return jit(**self.jitargs)(pyfunc)

    @contextlib.contextmanager
    def silence_numpy_warnings(self):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Implicitly casting between incompatible kinds', category=DeprecationWarning)
            yield

    def test_add_sub_timedelta(self):
        """
        Test `datetime64 + timedelta64` and `datetime64 - timedelta64`.
        """
        add = self.jit(add_usecase)
        sub = self.jit(sub_usecase)

        def check(a, b, expected):
            with self.silence_numpy_warnings():
                self.assertPreciseEqual(add(a, b), expected, (a, b))
                self.assertPreciseEqual(add(b, a), expected, (a, b))
                self.assertPreciseEqual(sub(a, -b), expected, (a, b))
                self.assertPreciseEqual(a + b, expected)
        check(DT('2014'), TD(2, 'Y'), DT('2016'))
        check(DT('2014'), TD(2, 'M'), DT('2014-03'))
        check(DT('2014'), TD(3, 'W'), DT('2014-01-16', 'W'))
        check(DT('2014'), TD(4, 'D'), DT('2014-01-05'))
        check(DT('2000'), TD(365, 'D'), DT('2000-12-31'))
        check(DT('2014-02'), TD(2, 'Y'), DT('2016-02'))
        check(DT('2014-02'), TD(2, 'M'), DT('2014-04'))
        check(DT('2014-02'), TD(2, 'D'), DT('2014-02-03'))
        check(DT('2014-01-07', 'W'), TD(2, 'W'), DT('2014-01-16', 'W'))
        check(DT('2014-02-02'), TD(27, 'D'), DT('2014-03-01'))
        check(DT('2012-02-02'), TD(27, 'D'), DT('2012-02-29'))
        check(DT('2012-02-02'), TD(2, 'W'), DT('2012-02-16'))
        check(DT('2000-01-01T01:02:03Z'), TD(2, 'h'), DT('2000-01-01T03:02:03Z'))
        check(DT('2000-01-01T01:02:03Z'), TD(2, 'ms'), DT('2000-01-01T01:02:03.002Z'))
        for dt_str in ('600', '601', '604', '801', '1900', '1904', '2200', '2300', '2304', '2400', '6001'):
            for dt_suffix in ('', '-01', '-12'):
                dt = DT(dt_str + dt_suffix)
                for td in [TD(2, 'D'), TD(2, 'W'), TD(100, 'D'), TD(10000, 'D'), TD(-100, 'D'), TD(-10000, 'D'), TD(100, 'W'), TD(10000, 'W'), TD(-100, 'W'), TD(-10000, 'W'), TD(100, 'M'), TD(10000, 'M'), TD(-100, 'M'), TD(-10000, 'M')]:
                    self.assertEqual(add(dt, td), dt + td, (dt, td))
                    self.assertEqual(add(td, dt), dt + td, (dt, td))
                    self.assertEqual(sub(dt, -td), dt + td, (dt, td))
        check(DT('NaT'), TD(2), DT('NaT'))
        check(DT('NaT', 's'), TD(2, 'h'), DT('NaT', 's'))
        check(DT('NaT', 's'), TD(2, 'ms'), DT('NaT', 'ms'))
        check(DT('2014'), TD('NaT', 'W'), DT('NaT', 'W'))
        check(DT('2014-01-01'), TD('NaT', 'W'), DT('NaT', 'D'))
        check(DT('NaT', 's'), TD('NaT', 'ms'), DT('NaT', 'ms'))
        for f in (add, sub):
            with self.assertRaises((TypeError, TypingError)):
                f(DT(1, '2014-01-01'), TD(1, 'Y'))
            with self.assertRaises((TypeError, TypingError)):
                f(DT(1, '2014-01-01'), TD(1, 'M'))

    def datetime_samples(self):
        dt_years = ['600', '601', '604', '1968', '1969', '1973', '2000', '2004', '2005', '2100', '2400', '2401']
        dt_suffixes = ['', '-01', '-12', '-02-28', '-12-31', '-01-05T12:30:56Z', '-01-05T12:30:56.008Z']
        dts = [DT(a + b) for a, b in itertools.product(dt_years, dt_suffixes)]
        dts += [DT(s, 'W') for s in dt_years]
        return dts

    def test_datetime_difference(self):
        """
        Test `datetime64 - datetime64`.
        """
        sub = self.jit(sub_usecase)

        def check(a, b, expected=None):
            with self.silence_numpy_warnings():
                self.assertPreciseEqual(sub(a, b), a - b, (a, b))
                self.assertPreciseEqual(sub(b, a), b - a, (a, b))
                self.assertPreciseEqual(a - b, expected)
        check(DT('2014'), DT('2017'), TD(-3, 'Y'))
        check(DT('2014-02'), DT('2017-01'), TD(-35, 'M'))
        check(DT('2014-02-28'), DT('2015-03-01'), TD(-366, 'D'))
        check(DT('NaT', 'M'), DT('2000'), TD('NaT', 'M'))
        check(DT('NaT', 'M'), DT('2000-01-01'), TD('NaT', 'D'))
        check(DT('NaT'), DT('NaT'), TD('NaT'))
        with self.silence_numpy_warnings():
            dts = self.datetime_samples()
            for a, b in itertools.product(dts, dts):
                if not npdatetime_helpers.same_kind(value_unit(a), value_unit(b)):
                    continue
                self.assertPreciseEqual(sub(a, b), a - b, (a, b))

    def test_comparisons(self):
        eq = self.jit(eq_usecase)
        ne = self.jit(ne_usecase)
        lt = self.jit(lt_usecase)
        le = self.jit(le_usecase)
        gt = self.jit(gt_usecase)
        ge = self.jit(ge_usecase)

        def check_eq(a, b, expected):
            expected_val = expected
            not_expected_val = not expected
            if np.isnat(a) or np.isnat(b):
                expected_val = False
                not_expected_val = True
                self.assertFalse(le(a, b), (a, b))
                self.assertFalse(ge(a, b), (a, b))
                self.assertFalse(le(b, a), (a, b))
                self.assertFalse(ge(b, a), (a, b))
                self.assertFalse(lt(a, b), (a, b))
                self.assertFalse(gt(a, b), (a, b))
                self.assertFalse(lt(b, a), (a, b))
                self.assertFalse(gt(b, a), (a, b))
            with self.silence_numpy_warnings():
                self.assertPreciseEqual(eq(a, b), expected_val, (a, b, expected))
                self.assertPreciseEqual(eq(b, a), expected_val, (a, b, expected))
                self.assertPreciseEqual(ne(a, b), not_expected_val, (a, b, expected))
                self.assertPreciseEqual(ne(b, a), not_expected_val, (a, b, expected))
                if expected_val:
                    self.assertTrue(le(a, b), (a, b))
                    self.assertTrue(ge(a, b), (a, b))
                    self.assertTrue(le(b, a), (a, b))
                    self.assertTrue(ge(b, a), (a, b))
                    self.assertFalse(lt(a, b), (a, b))
                    self.assertFalse(gt(a, b), (a, b))
                    self.assertFalse(lt(b, a), (a, b))
                    self.assertFalse(gt(b, a), (a, b))
                self.assertPreciseEqual(a == b, expected_val)

        def check_lt(a, b, expected):
            expected_val = expected
            not_expected_val = not expected
            if np.isnat(a) or np.isnat(b):
                expected_val = False
                not_expected_val = False
            with self.silence_numpy_warnings():
                lt = self.jit(lt_usecase)
                self.assertPreciseEqual(lt(a, b), expected_val, (a, b, expected))
                self.assertPreciseEqual(gt(b, a), expected_val, (a, b, expected))
                self.assertPreciseEqual(ge(a, b), not_expected_val, (a, b, expected))
                self.assertPreciseEqual(le(b, a), not_expected_val, (a, b, expected))
                if expected_val:
                    check_eq(a, b, False)
                self.assertPreciseEqual(a < b, expected_val)
        check_eq(DT('2014'), DT('2017'), False)
        check_eq(DT('2014'), DT('2014-01'), True)
        check_eq(DT('2014'), DT('2014-01-01'), True)
        check_eq(DT('2014'), DT('2014-01-01', 'W'), True)
        check_eq(DT('2014-01'), DT('2014-01-01', 'W'), True)
        check_eq(DT('2014-01-01'), DT('2014-01-01', 'W'), False)
        check_eq(DT('2014-01-02'), DT('2014-01-06', 'W'), True)
        check_eq(DT('2014-01-01T00:01:00Z', 's'), DT('2014-01-01T00:01Z', 'm'), True)
        check_eq(DT('2014-01-01T00:01:01Z', 's'), DT('2014-01-01T00:01Z', 'm'), False)
        check_lt(DT('NaT', 'Y'), DT('2017'), True)
        check_eq(DT('NaT'), DT('NaT'), True)
        dts = self.datetime_samples()
        for a in dts:
            a_unit = a.dtype.str.split('[')[1][:-1]
            i = all_units.index(a_unit)
            units = all_units[i:i + 6]
            for unit in units:
                b = a.astype('M8[%s]' % unit)
                if not npdatetime_helpers.same_kind(value_unit(a), value_unit(b)):
                    continue
                check_eq(a, b, True)
                check_lt(a, b + np.timedelta64(1, unit), True)
                check_lt(b - np.timedelta64(1, unit), a, True)

    def _test_min_max(self, usecase):
        f = self.jit(usecase)

        def check(a, b):
            self.assertPreciseEqual(f(a, b), usecase(a, b))
        for cases in ((DT(0, 'ns'), DT(1, 'ns'), DT(2, 'ns'), DT('NaT', 'ns')), (DT(0, 's'), DT(1, 's'), DT(2, 's'), DT('NaT', 's'))):
            for a, b in itertools.product(cases, cases):
                check(a, b)

    def test_min(self):
        self._test_min_max(min_usecase)

    def test_max(self):
        self._test_min_max(max_usecase)