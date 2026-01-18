import sys
import os.path
import re
import subprocess
import numpy as np
from numba.tests.support import capture_cache_log
from numba.tests.test_caching import BaseCacheTest
from numba.core import config
import unittest
class TestGUfuncCacheTest(UfuncCacheTest):

    def test_filename_prefix(self):
        mod = self.import_module()
        usecase = getattr(mod, 'direct_gufunc_cache_usecase')
        with capture_cache_log() as out:
            usecase()
        cachelog = out.getvalue()
        fmt1 = _fix_raw_path('/__pycache__/guf-{}')
        prefixed = re.findall(fmt1.format(self.modname), cachelog)
        fmt2 = _fix_raw_path('/__pycache__/{}')
        normal = re.findall(fmt2.format(self.modname), cachelog)
        self.assertGreater(len(normal), 2)
        self.assertEqual(len(normal), len(prefixed))

    def test_direct_gufunc_cache(self, **kwargs):
        new_ufunc, cached_ufunc = self.check_ufunc_cache('direct_gufunc_cache_usecase', n_overloads=2 + 2, **kwargs)
        inp = np.random.random(10).astype(np.float64)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))
        inp = np.arange(10, dtype=np.intp)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))

    def test_direct_gufunc_cache_objmode(self):
        self.test_direct_gufunc_cache(forceobj=True)

    def test_direct_gufunc_cache_parallel(self):
        self.test_direct_gufunc_cache(target='parallel')

    def test_indirect_gufunc_cache(self, **kwargs):
        new_ufunc, cached_ufunc = self.check_ufunc_cache('indirect_gufunc_cache_usecase', n_overloads=3, **kwargs)
        inp = np.random.random(10).astype(np.float64)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))
        inp = np.arange(10, dtype=np.intp)
        np.testing.assert_equal(new_ufunc(inp), cached_ufunc(inp))

    def test_indirect_gufunc_cache_parallel(self, **kwargs):
        self.test_indirect_gufunc_cache(target='parallel')