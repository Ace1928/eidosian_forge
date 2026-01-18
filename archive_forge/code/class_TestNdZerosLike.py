import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
class TestNdZerosLike(TestNdEmptyLike):

    def setUp(self):
        super(TestNdZerosLike, self).setUp()
        self.pyfunc = np.zeros_like

    def check_result_value(self, ret, expected):
        np.testing.assert_equal(ret, expected)

    def test_like_structured(self):
        super(TestNdZerosLike, self).test_like_structured()

    def test_like_dtype_structured(self):
        super(TestNdZerosLike, self).test_like_dtype_structured()