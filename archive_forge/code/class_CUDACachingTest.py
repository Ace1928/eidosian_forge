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
class CUDACachingTest(SerialMixin, DispatcherCacheUsecasesTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, 'cache_usecases.py')
    modname = 'cuda_caching_test_fodder'

    def setUp(self):
        DispatcherCacheUsecasesTest.setUp(self)
        CUDATestCase.setUp(self)

    def tearDown(self):
        CUDATestCase.tearDown(self)
        DispatcherCacheUsecasesTest.tearDown(self)

    def test_caching(self):
        self.check_pycache(0)
        mod = self.import_module()
        self.check_pycache(0)
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(2)
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        self.check_pycache(3)
        self.check_hits(f.func, 0, 2)
        f = mod.record_return_aligned
        rec = f(mod.aligned_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))
        f = mod.record_return_packed
        rec = f(mod.packed_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))
        self.check_pycache(6)
        self.check_hits(f.func, 0, 2)
        self.run_in_separate_process()

    def test_no_caching(self):
        mod = self.import_module()
        f = mod.add_nocache_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(0)

    def test_many_locals(self):
        self.check_pycache(0)
        mod = self.import_module()
        f = mod.many_locals
        f[1, 1]()
        self.check_pycache(2)

    def test_closure(self):
        mod = self.import_module()
        with warnings.catch_warnings():
            warnings.simplefilter('error', NumbaWarning)
            f = mod.closure1
            self.assertPreciseEqual(f(3), 6)
            f = mod.closure2
            self.assertPreciseEqual(f(3), 8)
            f = mod.closure3
            self.assertPreciseEqual(f(3), 10)
            f = mod.closure4
            self.assertPreciseEqual(f(3), 12)
            self.check_pycache(5)

    def test_cache_reuse(self):
        mod = self.import_module()
        mod.add_usecase(2, 3)
        mod.add_usecase(2.5, 3.5)
        mod.outer_uncached(2, 3)
        mod.outer(2, 3)
        mod.record_return_packed(mod.packed_arr, 0)
        mod.record_return_aligned(mod.aligned_arr, 1)
        mod.simple_usecase_caller(2)
        mtimes = self.get_cache_mtimes()
        self.check_hits(mod.add_usecase.func, 0, 2)
        mod2 = self.import_module()
        self.assertIsNot(mod, mod2)
        f = mod2.add_usecase
        f(2, 3)
        self.check_hits(f.func, 1, 0)
        f(2.5, 3.5)
        self.check_hits(f.func, 2, 0)
        self.assertEqual(self.get_cache_mtimes(), mtimes)
        self.run_in_separate_process()
        self.assertEqual(self.get_cache_mtimes(), mtimes)

    def test_cache_invalidate(self):
        mod = self.import_module()
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        with open(self.modfile, 'a') as f:
            f.write('\nZ = 10\n')
        mod = self.import_module()
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 15)

    def test_recompile(self):
        mod = self.import_module()
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        mod = self.import_module()
        f = mod.add_usecase
        mod.Z = 10
        self.assertPreciseEqual(f(2, 3), 6)
        f.func.recompile()
        self.assertPreciseEqual(f(2, 3), 15)
        mod = self.import_module()
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 15)

    def test_same_names(self):
        mod = self.import_module()
        f = mod.renamed_function1
        self.assertPreciseEqual(f(2), 4)
        f = mod.renamed_function2
        self.assertPreciseEqual(f(2), 8)

    @skip_unless_cc_60
    @skip_if_cudadevrt_missing
    @skip_if_mvc_enabled('CG not supported with MVC')
    def test_cache_cg(self):
        self.check_pycache(0)
        mod = self.import_module()
        self.check_pycache(0)
        mod.cg_usecase(0)
        self.check_pycache(2)
        self.run_in_separate_process()

    def _test_pycache_fallback(self):
        """
        With a disabled __pycache__, test there is a working fallback
        (e.g. on the user-wide cache dir)
        """
        mod = self.import_module()
        f = mod.add_usecase
        self.addCleanup(shutil.rmtree, f.func.stats.cache_path, ignore_errors=True)
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_hits(f.func, 0, 1)
        mod2 = self.import_module()
        f = mod2.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_hits(f.func, 1, 0)
        self.check_pycache(0)

    @skip_bad_access
    @unittest.skipIf(os.name == 'nt', 'cannot easily make a directory read-only on Windows')
    def test_non_creatable_pycache(self):
        old_perms = os.stat(self.tempdir).st_mode
        os.chmod(self.tempdir, 320)
        self.addCleanup(os.chmod, self.tempdir, old_perms)
        self._test_pycache_fallback()

    @skip_bad_access
    @unittest.skipIf(os.name == 'nt', 'cannot easily make a directory read-only on Windows')
    def test_non_writable_pycache(self):
        pycache = os.path.join(self.tempdir, '__pycache__')
        os.mkdir(pycache)
        old_perms = os.stat(pycache).st_mode
        os.chmod(pycache, 320)
        self.addCleanup(os.chmod, pycache, old_perms)
        self._test_pycache_fallback()

    def test_cannot_cache_linking_libraries(self):
        link = str(test_data_dir / 'jitlink.ptx')
        msg = 'Cannot pickle CUDACodeLibrary with linking files'
        with self.assertRaisesRegex(RuntimeError, msg):

            @cuda.jit('void()', cache=True, link=[link])
            def f():
                pass