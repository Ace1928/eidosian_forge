import os.path
import subprocess
import sys
import numpy as np
from numba.tests.support import skip_parfors_unsupported
from .test_caching import DispatcherCacheUsecasesTest
@skip_parfors_unsupported
class TestParforsCache(DispatcherCacheUsecasesTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, 'parfors_cache_usecases.py')
    modname = 'parfors_caching_test_fodder'

    def run_test(self, fname, num_funcs=1):
        mod = self.import_module()
        self.check_pycache(0)
        f = getattr(mod, fname)
        ary = np.ones(10)
        np.testing.assert_allclose(f(ary), f.py_func(ary))
        dynamic_globals = [cres.library.has_dynamic_globals for cres in f.overloads.values()]
        [cres] = f.overloads.values()
        self.assertEqual(dynamic_globals, [False])
        self.check_pycache(num_funcs * 2)
        self.run_in_separate_process()

    def test_arrayexprs(self):
        f = 'arrayexprs_case'
        self.run_test(f)

    def test_prange(self):
        f = 'prange_case'
        self.run_test(f)

    def test_caller(self):
        f = 'caller_case'
        self.run_test(f, num_funcs=3)