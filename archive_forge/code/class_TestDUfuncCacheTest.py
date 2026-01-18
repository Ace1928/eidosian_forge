import sys
import os.path
import re
import subprocess
import numpy as np
from numba.tests.support import capture_cache_log
from numba.tests.test_caching import BaseCacheTest
from numba.core import config
import unittest
class TestDUfuncCacheTest(UfuncCacheTest):

    def check_dufunc_usecase(self, usecase_name):
        mod = self.import_module()
        usecase = getattr(mod, usecase_name)
        with capture_cache_log() as out:
            ufunc = usecase()
        self.check_cache_saved(out.getvalue(), count=0)
        with capture_cache_log() as out:
            ufunc(np.arange(10))
        self.check_cache_saved(out.getvalue(), count=1)
        self.check_cache_loaded(out.getvalue(), count=0)
        with capture_cache_log() as out:
            ufunc = usecase()
            ufunc(np.arange(10))
        self.check_cache_loaded(out.getvalue(), count=1)

    def test_direct_dufunc_cache(self):
        self.check_dufunc_usecase('direct_dufunc_cache_usecase')

    def test_indirect_dufunc_cache(self):
        self.check_dufunc_usecase('indirect_dufunc_cache_usecase')