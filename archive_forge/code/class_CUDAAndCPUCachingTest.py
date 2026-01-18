import multiprocessing
import os
import shutil
import unittest
import warnings
from numba import cuda
from numba.core.errors import NumbaWarning
from numba.cuda.testing import (CUDATestCase, skip_on_cudasim,
from numba.tests.support import SerialMixin
from numba.tests.test_caching import (DispatcherCacheUsecasesTest,
@skip_on_cudasim('Simulator does not implement caching')
class CUDAAndCPUCachingTest(SerialMixin, DispatcherCacheUsecasesTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, 'cache_with_cpu_usecases.py')
    modname = 'cuda_and_cpu_caching_test_fodder'

    def setUp(self):
        DispatcherCacheUsecasesTest.setUp(self)
        CUDATestCase.setUp(self)

    def tearDown(self):
        CUDATestCase.tearDown(self)
        DispatcherCacheUsecasesTest.tearDown(self)

    def test_cpu_and_cuda_targets(self):
        self.check_pycache(0)
        mod = self.import_module()
        self.check_pycache(0)
        f_cpu = mod.assign_cpu
        f_cuda = mod.assign_cuda
        self.assertPreciseEqual(f_cpu(5), 5)
        self.check_pycache(2)
        self.assertPreciseEqual(f_cuda(5), 5)
        self.check_pycache(3)
        self.check_hits(f_cpu.func, 0, 1)
        self.check_hits(f_cuda.func, 0, 1)
        self.assertPreciseEqual(f_cpu(5.5), 5.5)
        self.check_pycache(4)
        self.assertPreciseEqual(f_cuda(5.5), 5.5)
        self.check_pycache(5)
        self.check_hits(f_cpu.func, 0, 2)
        self.check_hits(f_cuda.func, 0, 2)

    def test_cpu_and_cuda_reuse(self):
        mod = self.import_module()
        mod.assign_cpu(5)
        mod.assign_cpu(5.5)
        mod.assign_cuda(5)
        mod.assign_cuda(5.5)
        mtimes = self.get_cache_mtimes()
        self.check_hits(mod.assign_cpu.func, 0, 2)
        self.check_hits(mod.assign_cuda.func, 0, 2)
        mod2 = self.import_module()
        self.assertIsNot(mod, mod2)
        f_cpu = mod2.assign_cpu
        f_cuda = mod2.assign_cuda
        f_cpu(2)
        self.check_hits(f_cpu.func, 1, 0)
        f_cpu(2.5)
        self.check_hits(f_cpu.func, 2, 0)
        f_cuda(2)
        self.check_hits(f_cuda.func, 1, 0)
        f_cuda(2.5)
        self.check_hits(f_cuda.func, 2, 0)
        self.assertEqual(self.get_cache_mtimes(), mtimes)
        self.run_in_separate_process()
        self.assertEqual(self.get_cache_mtimes(), mtimes)