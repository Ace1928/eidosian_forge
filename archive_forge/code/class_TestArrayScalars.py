import sys
from itertools import product
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaNotImplementedError
from numba.tests.support import TestCase
from numba.tests.enum_usecases import Shake, RequestError
from numba.np import numpy_support
class TestArrayScalars(ValueTypingTestBase, TestCase):

    def test_number_values(self):
        """
        Test map_arrayscalar_type() with scalar number values.
        """
        self.check_number_values(numpy_support.map_arrayscalar_type)

    def test_datetime_values(self):
        """
        Test map_arrayscalar_type() with np.datetime64 values.
        """
        f = numpy_support.map_arrayscalar_type
        self.check_datetime_values(f)
        t = np.datetime64('2014', '10Y')
        with self.assertRaises(NotImplementedError):
            f(t)

    def test_timedelta_values(self):
        """
        Test map_arrayscalar_type() with np.timedelta64 values.
        """
        f = numpy_support.map_arrayscalar_type
        self.check_timedelta_values(f)
        t = np.timedelta64(10, '10Y')
        with self.assertRaises(NotImplementedError):
            f(t)