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
class TestMetadataScalingFactor(TestCase):
    """
    Tests than non-1 scaling factors are not supported in datetime64
    and timedelta64 dtypes.
    """

    def test_datetime(self, jitargs={'forceobj': True}):
        eq = jit(**jitargs)(eq_usecase)
        self.assertTrue(eq(DT('2014', '10Y'), DT('2010')))

    def test_datetime_npm(self):
        with self.assertTypingError():
            self.test_datetime(jitargs={'nopython': True})

    def test_timedelta(self, jitargs={'forceobj': True}):
        eq = jit(**jitargs)(eq_usecase)
        self.assertTrue(eq(TD(2, '10Y'), TD(20, 'Y')))

    def test_timedelta_npm(self):
        with self.assertTypingError():
            self.test_timedelta(jitargs={'nopython': True})