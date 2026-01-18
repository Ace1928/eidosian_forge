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
class TestRandomChoice(BaseTest):
    """
    Test np.random.choice.
    """

    def _check_results(self, pop, res, replace=True):
        """
        Check basic expectations about a batch of samples.
        """
        spop = set(pop)
        sres = set(res)
        self.assertLessEqual(sres, spop)
        self.assertNotEqual(sorted(res), list(res))
        if replace:
            self.assertLess(len(sres), len(res), res)
        else:
            self.assertEqual(len(sres), len(res), res)

    def _check_dist(self, pop, samples):
        """
        Check distribution of some samples.
        """
        self.assertGreaterEqual(len(samples), len(pop) * 100)
        expected_frequency = len(samples) / len(pop)
        c = collections.Counter(samples)
        for value in pop:
            n = c[value]
            self.assertGreaterEqual(n, expected_frequency * 0.5)
            self.assertLessEqual(n, expected_frequency * 2.0)

    def _accumulate_array_results(self, func, nresults):
        """
        Accumulate array results produced by *func* until they reach
        *nresults* elements.
        """
        res = []
        while len(res) < nresults:
            res += list(func().flat)
        return res[:nresults]

    def _check_choice_1(self, a, pop):
        """
        Check choice(a) against pop.
        """
        cfunc = jit(nopython=True)(numpy_choice1)
        n = len(pop)
        res = [cfunc(a) for i in range(n)]
        self._check_results(pop, res)
        dist = [cfunc(a) for i in range(n * 100)]
        self._check_dist(pop, dist)

    def test_choice_scalar_1(self):
        """
        Test choice(int)
        """
        n = 50
        pop = list(range(n))
        self._check_choice_1(n, pop)

    def test_choice_array_1(self):
        """
        Test choice(array)
        """
        pop = np.arange(50) * 2 + 100
        self._check_choice_1(pop, pop)

    def _check_array_results(self, func, pop, replace=True):
        """
        Check array results produced by *func* and their distribution.
        """
        n = len(pop)
        res = list(func().flat)
        self._check_results(pop, res, replace)
        dist = self._accumulate_array_results(func, n * 100)
        self._check_dist(pop, dist)

    def _check_choice_2(self, a, pop):
        """
        Check choice(a, size) against pop.
        """
        cfunc = jit(nopython=True)(numpy_choice2)
        n = len(pop)
        sizes = [n - 10, (3, (n - 1) // 3), n * 10]
        for size in sizes:
            res = cfunc(a, size)
            expected_shape = size if isinstance(size, tuple) else (size,)
            self.assertEqual(res.shape, expected_shape)
            self._check_array_results(lambda: cfunc(a, size), pop)

    def test_choice_scalar_2(self):
        """
        Test choice(int, size)
        """
        n = 50
        pop = np.arange(n)
        self._check_choice_2(n, pop)

    def test_choice_array_2(self):
        """
        Test choice(array, size)
        """
        pop = np.arange(50) * 2 + 100
        self._check_choice_2(pop, pop)

    def _check_choice_3(self, a, pop):
        """
        Check choice(a, size, replace) against pop.
        """
        cfunc = jit(nopython=True)(numpy_choice3)
        n = len(pop)
        sizes = [n - 10, (3, (n - 1) // 3)]
        replaces = [True, False]
        for size in sizes:
            for replace in [True, False]:
                res = cfunc(a, size, replace)
                expected_shape = size if isinstance(size, tuple) else (size,)
                self.assertEqual(res.shape, expected_shape)
        for size in sizes:
            self._check_array_results(lambda: cfunc(a, size, True), pop)
        for size in sizes:
            self._check_array_results(lambda: cfunc(a, size, False), pop, False)
        for size in [n + 1, (3, n // 3 + 1)]:
            with self.assertRaises(ValueError):
                cfunc(a, size, False)

    def test_choice_scalar_3(self):
        """
        Test choice(int, size, replace)
        """
        n = 50
        pop = np.arange(n)
        self._check_choice_3(n, pop)

    def test_choice_array_3(self):
        """
        Test choice(array, size, replace)
        """
        pop = np.arange(50) * 2 + 100
        self._check_choice_3(pop, pop)

    def test_choice_follows_seed(self):

        @jit(nopython=True)
        def numba_rands(n_to_return, choice_array):
            np.random.seed(1337)
            out = np.empty((n_to_return, 2), np.int32)
            for i in range(n_to_return):
                out[i] = np.random.choice(choice_array, 2, False)
            return out
        choice_array = np.random.randint(300, size=1000).astype(np.int32)
        tmp_np = choice_array.copy()
        expected = numba_rands.py_func(5, tmp_np)
        tmp_nb = choice_array.copy()
        got = numba_rands(5, tmp_nb)
        np.testing.assert_allclose(expected, got)
        np.testing.assert_allclose(choice_array, tmp_np)
        np.testing.assert_allclose(choice_array, tmp_nb)