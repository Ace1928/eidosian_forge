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
class TestCache(DispatcherCacheUsecasesTest):

    def test_caching(self):
        self.check_pycache(0)
        mod = self.import_module()
        self.check_pycache(0)
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(2)
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        self.check_pycache(3)
        self.check_hits(f, 0, 2)
        f = mod.add_objmode_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(5)
        self.assertPreciseEqual(f(2.5, 3), 6.5)
        self.check_pycache(6)
        self.check_hits(f, 0, 2)
        f = mod.record_return
        rec = f(mod.aligned_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))
        rec = f(mod.packed_arr, 1)
        self.assertPreciseEqual(tuple(rec), (2, 43.5))
        self.check_pycache(9)
        self.check_hits(f, 0, 2)
        self.run_in_separate_process()

    def test_caching_nrt_pruned(self):
        self.check_pycache(0)
        mod = self.import_module()
        self.check_pycache(0)
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(2)
        self.assertPreciseEqual(f(2, np.arange(3)), 2 + np.arange(3) + 1)
        self.check_pycache(3)
        self.check_hits(f, 0, 2)

    def test_inner_then_outer(self):
        mod = self.import_module()
        self.assertPreciseEqual(mod.inner(3, 2), 6)
        self.check_pycache(2)
        f = mod.outer_uncached
        self.assertPreciseEqual(f(3, 2), 2)
        self.check_pycache(2)
        mod = self.import_module()
        f = mod.outer_uncached
        self.assertPreciseEqual(f(3, 2), 2)
        self.check_pycache(2)
        f = mod.outer
        self.assertPreciseEqual(f(3, 2), 2)
        self.check_pycache(4)
        self.assertPreciseEqual(f(3.5, 2), 2.5)
        self.check_pycache(6)

    def test_outer_then_inner(self):
        mod = self.import_module()
        self.assertPreciseEqual(mod.outer(3, 2), 2)
        self.check_pycache(4)
        self.assertPreciseEqual(mod.outer_uncached(3, 2), 2)
        self.check_pycache(4)
        mod = self.import_module()
        f = mod.inner
        self.assertPreciseEqual(f(3, 2), 6)
        self.check_pycache(4)
        self.assertPreciseEqual(f(3.5, 2), 6.5)
        self.check_pycache(5)

    def test_no_caching(self):
        mod = self.import_module()
        f = mod.add_nocache_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_pycache(0)

    def test_looplifted(self):
        mod = self.import_module()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)
            f = mod.looplifted
            self.assertPreciseEqual(f(4), 6)
            self.check_pycache(0)
        self.assertEqual(len(w), 1)
        self.assertIn('Cannot cache compiled function "looplifted" as it uses lifted code', str(w[0].message))

    def test_big_array(self):
        mod = self.import_module()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', NumbaWarning)
            f = mod.use_big_array
            np.testing.assert_equal(f(), mod.biggie)
            self.check_pycache(0)
        self.assertEqual(len(w), 1)
        self.assertIn('Cannot cache compiled function "use_big_array" as it uses dynamic globals', str(w[0].message))

    def test_ctypes(self):
        mod = self.import_module()
        for f in [mod.use_c_sin, mod.use_c_sin_nest1, mod.use_c_sin_nest2]:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always', NumbaWarning)
                self.assertPreciseEqual(f(0.0), 0.0)
                self.check_pycache(0)
            self.assertEqual(len(w), 1)
            self.assertIn('Cannot cache compiled function "{}"'.format(f.__name__), str(w[0].message))

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

    def test_first_class_function(self):
        mod = self.import_module()
        f = mod.first_class_function_usecase
        self.assertEqual(f(mod.first_class_function_mul, 1), 1)
        self.assertEqual(f(mod.first_class_function_mul, 10), 100)
        self.assertEqual(f(mod.first_class_function_add, 1), 2)
        self.assertEqual(f(mod.first_class_function_add, 10), 20)
        self.check_pycache(7)

    def test_cache_reuse(self):
        mod = self.import_module()
        mod.add_usecase(2, 3)
        mod.add_usecase(2.5, 3.5)
        mod.add_objmode_usecase(2, 3)
        mod.outer_uncached(2, 3)
        mod.outer(2, 3)
        mod.record_return(mod.packed_arr, 0)
        mod.record_return(mod.aligned_arr, 1)
        mtimes = self.get_cache_mtimes()
        self.check_hits(mod.add_usecase, 0, 2)
        mod2 = self.import_module()
        self.assertIsNot(mod, mod2)
        f = mod2.add_usecase
        f(2, 3)
        self.check_hits(f, 1, 0)
        f(2.5, 3.5)
        self.check_hits(f, 2, 0)
        f = mod2.add_objmode_usecase
        f(2, 3)
        self.check_hits(f, 1, 0)
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
        f = mod.add_objmode_usecase
        self.assertPreciseEqual(f(2, 3), 15)

    def test_recompile(self):
        mod = self.import_module()
        f = mod.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        mod = self.import_module()
        f = mod.add_usecase
        mod.Z = 10
        self.assertPreciseEqual(f(2, 3), 6)
        f.recompile()
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

    def test_frozen(self):
        from .dummy_module import function
        old_code = function.__code__
        code_obj = compile('pass', 'tests/dummy_module.py', 'exec')
        try:
            function.__code__ = code_obj
            source = inspect.getfile(function)
            locator = _UserWideCacheLocator.from_function(function, source)
            self.assertIsNone(locator)
            sys.frozen = True
            locator = _UserWideCacheLocator.from_function(function, source)
            self.assertIsInstance(locator, _UserWideCacheLocator)
        finally:
            function.__code__ = old_code
            del sys.frozen

    def _test_pycache_fallback(self):
        """
        With a disabled __pycache__, test there is a working fallback
        (e.g. on the user-wide cache dir)
        """
        mod = self.import_module()
        f = mod.add_usecase
        self.addCleanup(shutil.rmtree, f.stats.cache_path, ignore_errors=True)
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_hits(f, 0, 1)
        mod2 = self.import_module()
        f = mod2.add_usecase
        self.assertPreciseEqual(f(2, 3), 6)
        self.check_hits(f, 1, 0)
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

    def test_ipython(self):
        base_cmd = [sys.executable, '-m', 'IPython']
        base_cmd += ['--quiet', '--quick', '--no-banner', '--colors=NoColor']
        try:
            ver = subprocess.check_output(base_cmd + ['--version'])
        except subprocess.CalledProcessError as e:
            self.skipTest('ipython not available: return code %d' % e.returncode)
        ver = ver.strip().decode()
        inputfn = os.path.join(self.tempdir, 'ipython_cache_usecase.txt')
        with open(inputfn, 'w') as f:
            f.write('\n                import os\n                import sys\n\n                from numba import jit\n\n                # IPython 5 does not support multiline input if stdin isn\'t\n                # a tty (https://github.com/ipython/ipython/issues/9752)\n                f = jit(cache=True)(lambda: 42)\n\n                res = f()\n                # IPython writes on stdout, so use stderr instead\n                sys.stderr.write(u"cache hits = %d\\n" % f.stats.cache_hits[()])\n\n                # IPython hijacks sys.exit(), bypass it\n                sys.stdout.flush()\n                sys.stderr.flush()\n                os._exit(res)\n                ')

        def execute_with_input():
            with open(inputfn, 'rb') as stdin:
                p = subprocess.Popen(base_cmd, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                out, err = p.communicate()
                if p.returncode != 42:
                    self.fail('unexpected return code %d\n-- stdout:\n%s\n-- stderr:\n%s\n' % (p.returncode, out, err))
                return err
        execute_with_input()
        err = execute_with_input()
        self.assertIn('cache hits = 1', err.strip())

    @unittest.skipIf(ipykernel is None or ipykernel.version_info[0] < 6, 'requires ipykernel >= 6')
    def test_ipykernel(self):
        base_cmd = [sys.executable, '-m', 'IPython']
        base_cmd += ['--quiet', '--quick', '--no-banner', '--colors=NoColor']
        try:
            ver = subprocess.check_output(base_cmd + ['--version'])
        except subprocess.CalledProcessError as e:
            self.skipTest('ipython not available: return code %d' % e.returncode)
        ver = ver.strip().decode()
        from ipykernel import compiler
        inputfn = compiler.get_tmp_directory()
        with open(inputfn, 'w') as f:
            f.write('\n                import os\n                import sys\n\n                from numba import jit\n\n                # IPython 5 does not support multiline input if stdin isn\'t\n                # a tty (https://github.com/ipython/ipython/issues/9752)\n                f = jit(cache=True)(lambda: 42)\n\n                res = f()\n                # IPython writes on stdout, so use stderr instead\n                sys.stderr.write(u"cache hits = %d\\n" % f.stats.cache_hits[()])\n\n                # IPython hijacks sys.exit(), bypass it\n                sys.stdout.flush()\n                sys.stderr.flush()\n                os._exit(res)\n                ')

        def execute_with_input():
            with open(inputfn, 'rb') as stdin:
                p = subprocess.Popen(base_cmd, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                out, err = p.communicate()
                if p.returncode != 42:
                    self.fail('unexpected return code %d\n-- stdout:\n%s\n-- stderr:\n%s\n' % (p.returncode, out, err))
                return err
        execute_with_input()
        err = execute_with_input()
        self.assertIn('cache hits = 1', err.strip())