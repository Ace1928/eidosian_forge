import inspect
import llvmlite.binding as ll
import multiprocessing
import numpy as np
import os
import stat
import shutil
import subprocess
import sys
import traceback
import unittest
import warnings
from numba import njit
from numba.core import codegen
from numba.core.caching import _UserWideCacheLocator
from numba.core.errors import NumbaWarning
from numba.parfors import parfor
from numba.tests.support import (
from numba import njit
from numba import njit
from file2 import function2
from numba import njit
@skip_parfors_unsupported
class TestSequentialParForsCache(DispatcherCacheUsecasesTest):

    def setUp(self):
        super(TestSequentialParForsCache, self).setUp()
        parfor.sequential_parfor_lowering = True

    def tearDown(self):
        super(TestSequentialParForsCache, self).tearDown()
        parfor.sequential_parfor_lowering = False

    def test_caching(self):
        mod = self.import_module()
        self.check_pycache(0)
        f = mod.parfor_usecase
        ary = np.ones(10)
        self.assertPreciseEqual(f(ary), ary * ary + ary)
        dynamic_globals = [cres.library.has_dynamic_globals for cres in f.overloads.values()]
        self.assertEqual(dynamic_globals, [False])
        self.check_pycache(2)