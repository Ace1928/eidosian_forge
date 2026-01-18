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
class TestTimedeltaArithmetic(TestCase):
    jitargs = dict(forceobj=True)

    def jit(self, pyfunc):
        return jit(**self.jitargs)(pyfunc)

    def test_add(self):
        f = self.jit(add_usecase)

        def check(a, b, expected):
            self.assertPreciseEqual(f(a, b), expected)
            self.assertPreciseEqual(f(b, a), expected)
        check(TD(1), TD(2), TD(3))
        check(TD(1, 's'), TD(2, 's'), TD(3, 's'))
        check(TD(1, 's'), TD(2, 'us'), TD(1000002, 'us'))
        check(TD(1, 'W'), TD(2, 'D'), TD(9, 'D'))
        check(TD('NaT'), TD(1), TD('NaT'))
        check(TD('NaT', 's'), TD(1, 'D'), TD('NaT', 's'))
        check(TD('NaT', 's'), TD(1, 'ms'), TD('NaT', 'ms'))
        with self.assertRaises((TypeError, TypingError)):
            f(TD(1, 'M'), TD(1, 'D'))

    def test_sub(self):
        f = self.jit(sub_usecase)

        def check(a, b, expected):
            self.assertPreciseEqual(f(a, b), expected)
            self.assertPreciseEqual(f(b, a), -expected)
        check(TD(3), TD(2), TD(1))
        check(TD(3, 's'), TD(2, 's'), TD(1, 's'))
        check(TD(3, 's'), TD(2, 'us'), TD(2999998, 'us'))
        check(TD(1, 'W'), TD(2, 'D'), TD(5, 'D'))
        check(TD('NaT'), TD(1), TD('NaT'))
        check(TD('NaT', 's'), TD(1, 'D'), TD('NaT', 's'))
        check(TD('NaT', 's'), TD(1, 'ms'), TD('NaT', 'ms'))
        with self.assertRaises((TypeError, TypingError)):
            f(TD(1, 'M'), TD(1, 'D'))

    def test_mul(self):
        f = self.jit(mul_usecase)

        def check(a, b, expected):
            self.assertPreciseEqual(f(a, b), expected)
            self.assertPreciseEqual(f(b, a), expected)
        check(TD(3), np.uint32(2), TD(6))
        check(TD(3), 2, TD(6))
        check(TD(3, 'ps'), 2, TD(6, 'ps'))
        check(TD('NaT', 'ps'), 2, TD('NaT', 'ps'))
        check(TD(7), 1.5, TD(10))
        check(TD(-7), 1.5, TD(-10))
        check(TD(7, 'ps'), -1.5, TD(-10, 'ps'))
        check(TD(-7), -1.5, TD(10))
        check(TD('NaT', 'ps'), -1.5, TD('NaT', 'ps'))
        check(TD(7, 'ps'), float('nan'), TD('NaT', 'ps'))
        check(TD(2 ** 62, 'ps'), 16, TD(0, 'ps'))

    def test_div(self):
        div = self.jit(div_usecase)
        floordiv = self.jit(floordiv_usecase)

        def check(a, b, expected):
            self.assertPreciseEqual(div(a, b), expected)
            self.assertPreciseEqual(floordiv(a, b), expected)
        check(TD(-3, 'ps'), np.uint32(2), TD(-1, 'ps'))
        check(TD(3), 2, TD(1))
        check(TD(-3, 'ps'), 2, TD(-1, 'ps'))
        check(TD('NaT', 'ps'), 2, TD('NaT', 'ps'))
        check(TD(3, 'ps'), 0, TD('NaT', 'ps'))
        check(TD('NaT', 'ps'), 0, TD('NaT', 'ps'))
        check(TD(7), 0.5, TD(14))
        check(TD(-7, 'ps'), 1.5, TD(-4, 'ps'))
        check(TD('NaT', 'ps'), 2.5, TD('NaT', 'ps'))
        check(TD(3, 'ps'), 0.0, TD('NaT', 'ps'))
        check(TD('NaT', 'ps'), 0.0, TD('NaT', 'ps'))
        check(TD(3, 'ps'), float('nan'), TD('NaT', 'ps'))
        check(TD('NaT', 'ps'), float('nan'), TD('NaT', 'ps'))

    def test_homogeneous_div(self):
        div = self.jit(div_usecase)

        def check(a, b, expected):
            self.assertPreciseEqual(div(a, b), expected)
        check(TD(7), TD(3), 7.0 / 3.0)
        check(TD(7, 'us'), TD(3, 'ms'), 7.0 / 3000.0)
        check(TD(7, 'ms'), TD(3, 'us'), 7000.0 / 3.0)
        check(TD(7), TD(0), float('+inf'))
        check(TD(-7), TD(0), float('-inf'))
        check(TD(0), TD(0), float('nan'))
        check(TD('nat'), TD(3), float('nan'))
        check(TD(3), TD('nat'), float('nan'))
        check(TD('nat'), TD(0), float('nan'))
        with self.assertRaises((TypeError, TypingError)):
            div(TD(1, 'M'), TD(1, 'D'))

    def test_eq_ne(self):
        eq = self.jit(eq_usecase)
        ne = self.jit(ne_usecase)

        def check(a, b, expected):
            expected_val = expected
            not_expected_val = not expected
            if np.isnat(a) or np.isnat(a):
                expected_val = False
                not_expected_val = True
            self.assertPreciseEqual(eq(a, b), expected_val)
            self.assertPreciseEqual(eq(b, a), expected_val)
            self.assertPreciseEqual(ne(a, b), not_expected_val)
            self.assertPreciseEqual(ne(b, a), not_expected_val)
        check(TD(1), TD(2), False)
        check(TD(1), TD(1), True)
        check(TD(1, 's'), TD(2, 's'), False)
        check(TD(1, 's'), TD(1, 's'), True)
        check(TD(2000, 's'), TD(2, 's'), False)
        check(TD(2000, 'ms'), TD(2, 's'), True)
        check(TD(1, 'Y'), TD(12, 'M'), True)
        check(TD('Nat'), TD('Nat'), True)
        check(TD('Nat', 'ms'), TD('Nat', 's'), True)
        check(TD('Nat'), TD(1), False)
        if numpy_version < (1, 25):
            check(TD(1, 'Y'), TD(365, 'D'), False)
            check(TD(1, 'Y'), TD(366, 'D'), False)
            check(TD('NaT', 'W'), TD('NaT', 'D'), True)
        else:
            with self.assertRaises((TypeError, TypingError)):
                eq(TD(1, 'Y'), TD(365, 'D'))
            with self.assertRaises((TypeError, TypingError)):
                ne(TD(1, 'Y'), TD(365, 'D'))

    def test_lt_ge(self):
        lt = self.jit(lt_usecase)
        ge = self.jit(ge_usecase)

        def check(a, b, expected):
            expected_val = expected
            not_expected_val = not expected
            if np.isnat(a) or np.isnat(a):
                expected_val = False
                not_expected_val = False
            self.assertPreciseEqual(lt(a, b), expected_val)
            self.assertPreciseEqual(ge(a, b), not_expected_val)
        check(TD(1), TD(2), True)
        check(TD(1), TD(1), False)
        check(TD(2), TD(1), False)
        check(TD(1, 's'), TD(2, 's'), True)
        check(TD(1, 's'), TD(1, 's'), False)
        check(TD(2, 's'), TD(1, 's'), False)
        check(TD(1, 'm'), TD(61, 's'), True)
        check(TD(1, 'm'), TD(60, 's'), False)
        check(TD('Nat'), TD('Nat'), False)
        check(TD('Nat', 'ms'), TD('Nat', 's'), False)
        check(TD('Nat'), TD(-2 ** 63 + 1), True)
        with self.assertRaises((TypeError, TypingError)):
            lt(TD(1, 'Y'), TD(365, 'D'))
        with self.assertRaises((TypeError, TypingError)):
            ge(TD(1, 'Y'), TD(365, 'D'))
        with self.assertRaises((TypeError, TypingError)):
            lt(TD('NaT', 'Y'), TD('NaT', 'D'))
        with self.assertRaises((TypeError, TypingError)):
            ge(TD('NaT', 'Y'), TD('NaT', 'D'))

    def test_le_gt(self):
        le = self.jit(le_usecase)
        gt = self.jit(gt_usecase)

        def check(a, b, expected):
            expected_val = expected
            not_expected_val = not expected
            if np.isnat(a) or np.isnat(a):
                expected_val = False
                not_expected_val = False
            self.assertPreciseEqual(le(a, b), expected_val)
            self.assertPreciseEqual(gt(a, b), not_expected_val)
        check(TD(1), TD(2), True)
        check(TD(1), TD(1), True)
        check(TD(2), TD(1), False)
        check(TD(1, 's'), TD(2, 's'), True)
        check(TD(1, 's'), TD(1, 's'), True)
        check(TD(2, 's'), TD(1, 's'), False)
        check(TD(1, 'm'), TD(61, 's'), True)
        check(TD(1, 'm'), TD(60, 's'), True)
        check(TD(1, 'm'), TD(59, 's'), False)
        check(TD('Nat'), TD('Nat'), True)
        check(TD('Nat', 'ms'), TD('Nat', 's'), True)
        check(TD('Nat'), TD(-2 ** 63 + 1), True)
        with self.assertRaises((TypeError, TypingError)):
            le(TD(1, 'Y'), TD(365, 'D'))
        with self.assertRaises((TypeError, TypingError)):
            gt(TD(1, 'Y'), TD(365, 'D'))
        with self.assertRaises((TypeError, TypingError)):
            le(TD('NaT', 'Y'), TD('NaT', 'D'))
        with self.assertRaises((TypeError, TypingError)):
            gt(TD('NaT', 'Y'), TD('NaT', 'D'))

    def test_pos(self):
        pos = self.jit(pos_usecase)

        def check(a):
            self.assertPreciseEqual(pos(a), +a)
        check(TD(3))
        check(TD(-4))
        check(TD(3, 'ms'))
        check(TD(-4, 'ms'))
        check(TD('NaT'))
        check(TD('NaT', 'ms'))

    def test_neg(self):
        neg = self.jit(neg_usecase)

        def check(a):
            self.assertPreciseEqual(neg(a), -a)
        check(TD(3))
        check(TD(-4))
        check(TD(3, 'ms'))
        check(TD(-4, 'ms'))
        check(TD('NaT'))
        check(TD('NaT', 'ms'))

    def test_abs(self):
        f = self.jit(abs_usecase)

        def check(a):
            self.assertPreciseEqual(f(a), abs(a))
        check(TD(3))
        check(TD(-4))
        check(TD(3, 'ms'))
        check(TD(-4, 'ms'))
        check(TD('NaT'))
        check(TD('NaT', 'ms'))

    def test_hash(self):
        f = self.jit(hash_usecase)

        def check(a):
            self.assertPreciseEqual(f(a), hash(a))
        TD_CASES = ((3,), (-4,), (3, 'ms'), (-4, 'ms'), (27, 'D'), (2, 'D'), (2, 'W'), (2, 'Y'), (3, 'W'), (365, 'D'), (10000, 'D'), (-10000, 'D'), ('NaT',), ('NaT', 'ms'), ('NaT', 'D'), (-1,))
        DT_CASES = (('2014',), ('2016',), ('2000',), ('2014-02',), ('2014-03',), ('2014-04',), ('2016-02',), ('2000-12-31',), ('2014-01-16',), ('2014-01-05',), ('2014-01-07',), ('2014-01-06',), ('2014-02-02',), ('2014-02-27',), ('2014-02-16',), ('2014-03-01',), ('2000-01-01T01:02:03.002Z',), ('2000-01-01T01:02:03Z',), ('NaT',))
        for case, typ in zip(TD_CASES + DT_CASES, (TD,) * len(TD_CASES) + (DT,) * len(TD_CASES)):
            check(typ(*case))

    def _test_min_max(self, usecase):
        f = self.jit(usecase)

        def check(a, b):
            self.assertPreciseEqual(f(a, b), usecase(a, b))
        for cases in ((TD(0), TD(1), TD(2), TD('NaT')), (TD(0, 's'), TD(1, 's'), TD(2, 's'), TD('NaT', 's'))):
            for a, b in itertools.product(cases, cases):
                check(a, b)

    def test_min(self):
        self._test_min_max(min_usecase)

    def test_max(self):
        self._test_min_max(max_usecase)