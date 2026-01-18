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
class TestMultiprocessCache(SerialMixin, DispatcherCacheUsecasesTest):
    _numba_parallel_test_ = False
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, 'cache_usecases.py')
    modname = 'cuda_mp_caching_test_fodder'

    def setUp(self):
        DispatcherCacheUsecasesTest.setUp(self)
        CUDATestCase.setUp(self)

    def tearDown(self):
        CUDATestCase.tearDown(self)
        DispatcherCacheUsecasesTest.tearDown(self)

    def test_multiprocessing(self):
        mod = self.import_module()
        f = mod.simple_usecase_caller
        n = 3
        try:
            ctx = multiprocessing.get_context('spawn')
        except AttributeError:
            ctx = multiprocessing
        pool = ctx.Pool(n, child_initializer)
        try:
            res = sum(pool.imap(f, range(n)))
        finally:
            pool.close()
        self.assertEqual(res, n * (n - 1) // 2)