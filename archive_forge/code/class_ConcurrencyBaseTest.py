import collections
import functools
import math
import multiprocessing
import os
import random
import subprocess
import sys
import threading
import itertools
from textwrap import dedent
import numpy as np
import unittest
import numba
from numba import jit, _helperlib, njit
from numba.core import types
from numba.tests.support import TestCase, compile_function, tag
from numba.core.errors import TypingError
class ConcurrencyBaseTest(TestCase):
    _extract_iterations = 100000

    def setUp(self):
        args = (42, self._get_output(1))
        py_extract_randomness(*args)
        np_extract_randomness(*args)

    def _get_output(self, size):
        return np.zeros(size, dtype=np.uint32)

    def check_output(self, out):
        """
        Check statistical properties of output.
        """
        expected_avg = 1 << 31
        expected_std = (1 << 32) / np.sqrt(12)
        rtol = 0.05
        np.testing.assert_allclose(out.mean(), expected_avg, rtol=rtol)
        np.testing.assert_allclose(out.std(), expected_std, rtol=rtol)

    def check_several_outputs(self, results, same_expected):
        for out in results:
            self.check_output(out)
        if same_expected:
            expected_distinct = 1
        else:
            expected_distinct = len(results)
        heads = {tuple(out[:5]) for out in results}
        tails = {tuple(out[-5:]) for out in results}
        sums = {out.sum() for out in results}
        self.assertEqual(len(heads), expected_distinct, heads)
        self.assertEqual(len(tails), expected_distinct, tails)
        self.assertEqual(len(sums), expected_distinct, sums)