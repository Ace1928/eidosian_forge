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
class TestModuleHelpers(TestCase):
    """
    Test the various helpers in numba.npdatetime_helpers.
    """

    def test_can_cast_timedelta(self):
        f = npdatetime_helpers.can_cast_timedelta_units
        for a, b in itertools.product(date_units, time_units):
            self.assertFalse(f(a, b), (a, b))
            self.assertFalse(f(b, a), (a, b))
        for unit in all_units:
            self.assertFalse(f(unit, ''))
            self.assertTrue(f('', unit))
        for unit in all_units + ('',):
            self.assertTrue(f(unit, unit))

        def check_units_group(group):
            for i, a in enumerate(group):
                for b in group[:i]:
                    self.assertTrue(f(b, a))
                    self.assertFalse(f(a, b))
        check_units_group(date_units)
        check_units_group(time_units)

    def test_timedelta_conversion(self):
        f = npdatetime_helpers.get_timedelta_conversion_factor
        for unit in all_units + ('',):
            self.assertEqual(f(unit, unit), 1)
        for unit in all_units:
            self.assertEqual(f('', unit), 1)
        for a, b in itertools.product(time_units, date_units):
            self.assertIs(f(a, b), None)
            self.assertIs(f(b, a), None)

        def check_units_group(group):
            for i, a in enumerate(group):
                for b in group[:i]:
                    self.assertGreater(f(b, a), 1, (b, a))
                    self.assertIs(f(a, b), None)
        check_units_group(date_units)
        check_units_group(time_units)
        self.assertEqual(f('Y', 'M'), 12)
        self.assertEqual(f('W', 'h'), 24 * 7)
        self.assertEqual(f('W', 'm'), 24 * 7 * 60)
        self.assertEqual(f('W', 'us'), 24 * 7 * 3600 * 1000 * 1000)

    def test_datetime_timedelta_scaling(self):
        f = npdatetime_helpers.get_datetime_timedelta_conversion

        def check_error(dt_unit, td_unit):
            with self.assertRaises(RuntimeError):
                f(dt_unit, td_unit)
        for dt_unit, td_unit in itertools.product(time_units, date_units):
            check_error(dt_unit, td_unit)
        for dt_unit, td_unit in itertools.product(time_units, time_units):
            f(dt_unit, td_unit)
        for dt_unit, td_unit in itertools.product(date_units, time_units):
            f(dt_unit, td_unit)
        for dt_unit, td_unit in itertools.product(date_units, date_units):
            f(dt_unit, td_unit)
        for unit in all_units:
            self.assertEqual(f(unit, unit), (unit, 1, 1))
            self.assertEqual(f(unit, ''), (unit, 1, 1))
            self.assertEqual(f('', unit), ('', 1, 1))
        self.assertEqual(f('', ''), ('', 1, 1))
        self.assertEqual(f('Y', 'M'), ('M', 12, 1))
        self.assertEqual(f('M', 'Y'), ('M', 1, 12))
        self.assertEqual(f('W', 'D'), ('D', 7, 1))
        self.assertEqual(f('D', 'W'), ('D', 1, 7))
        self.assertEqual(f('W', 's'), ('s', 7 * 24 * 3600, 1))
        self.assertEqual(f('s', 'W'), ('s', 1, 7 * 24 * 3600))
        self.assertEqual(f('s', 'as'), ('as', 1000 ** 6, 1))
        self.assertEqual(f('as', 's'), ('as', 1, 1000 ** 6))
        self.assertEqual(f('Y', 'D'), ('D', 97 + 400 * 365, 400))
        self.assertEqual(f('Y', 'W'), ('W', 97 + 400 * 365, 400 * 7))
        self.assertEqual(f('M', 'D'), ('D', 97 + 400 * 365, 400 * 12))
        self.assertEqual(f('M', 'W'), ('W', 97 + 400 * 365, 400 * 12 * 7))
        self.assertEqual(f('Y', 's'), ('s', (97 + 400 * 365) * 24 * 3600, 400))
        self.assertEqual(f('M', 's'), ('s', (97 + 400 * 365) * 24 * 3600, 400 * 12))

    def test_combine_datetime_timedelta_units(self):
        f = npdatetime_helpers.combine_datetime_timedelta_units
        for unit in all_units:
            self.assertEqual(f(unit, unit), unit)
            self.assertEqual(f('', unit), unit)
            self.assertEqual(f(unit, ''), unit)
        self.assertEqual(f('', ''), '')
        for dt_unit, td_unit in itertools.product(time_units, date_units):
            self.assertIs(f(dt_unit, td_unit), None)
        for dt_unit, td_unit in itertools.product(date_units, time_units):
            self.assertEqual(f(dt_unit, td_unit), td_unit)

    def test_same_kind(self):
        f = npdatetime_helpers.same_kind
        for u in all_units:
            self.assertTrue(f(u, u))
        A = ('Y', 'M', 'W', 'D')
        B = ('h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as')
        for a, b in itertools.product(A, A):
            self.assertTrue(f(a, b))
        for a, b in itertools.product(B, B):
            self.assertTrue(f(a, b))
        for a, b in itertools.product(A, B):
            self.assertFalse(f(a, b))
            self.assertFalse(f(b, a))