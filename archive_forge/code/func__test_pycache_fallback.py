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