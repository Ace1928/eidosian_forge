import numpy as np
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba import vectorize, guvectorize
from numba.np.ufunc import PyUFunc_One
from numba.np.ufunc.dufunc import DUFunc as UFuncBuilder
from numba.tests.support import tag, TestCase
from numba.core import config
import unittest
class TestUfuncBuildingJitDisabled(TestUfuncBuilding):

    def setUp(self):
        self.old_disable_jit = config.DISABLE_JIT
        config.DISABLE_JIT = False

    def tearDown(self):
        config.DISABLE_JIT = self.old_disable_jit