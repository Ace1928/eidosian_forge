import multiprocessing
import platform
import threading
import pickle
import weakref
from itertools import chain
from io import StringIO
import numpy as np
from numba import njit, jit, typeof, vectorize
from numba.core import types, errors
from numba import _dispatcher
from numba.tests.support import TestCase, captured_stdout
from numba.np.numpy_support import as_dtype
from numba.core.dispatcher import Dispatcher
from numba.extending import overload
from numba.tests.support import needs_lapack, SerialMixin
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
import unittest
class TestMultiprocessingDefaultParameters(SerialMixin, unittest.TestCase):

    def run_fc_multiproc(self, fc):
        try:
            ctx = multiprocessing.get_context('spawn')
        except AttributeError:
            ctx = multiprocessing
        for a in [1, 2, 3]:
            p = ctx.Process(target=_checker, args=(fc, a))
            p.start()
            p.join(_TEST_TIMEOUT)
            self.assertEqual(p.exitcode, 0)

    def test_int_def_param(self):
        """ Tests issue #4888"""
        self.run_fc_multiproc(add_y1)

    def test_none_def_param(self):
        """ Tests None as a default parameter"""
        self.run_fc_multiproc(add_func)

    def test_function_def_param(self):
        """ Tests a function as a default parameter"""
        self.run_fc_multiproc(add_func)