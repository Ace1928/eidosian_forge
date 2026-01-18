import os.path
import subprocess
import sys
import numpy as np
from numba.tests.support import skip_parfors_unsupported
from .test_caching import DispatcherCacheUsecasesTest
@skip_parfors_unsupported
class TestParforsCacheChangingThreads(DispatcherCacheUsecasesTest):
    here = os.path.dirname(__file__)
    usecases_file = os.path.join(here, 'parfors_cache_usecases.py')
    modname = 'parfors_caching_test_fodder'

    def run_in_separate_process(self, thread_count):
        code = 'if 1:\n            import sys\n\n            sys.path.insert(0, %(tempdir)r)\n            mod = __import__(%(modname)r)\n            mod.self_run()\n            ' % dict(tempdir=self.tempdir, modname=self.modname)
        new_env = {**os.environ, 'NUMBA_NUM_THREADS': str(thread_count)}
        popen = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=new_env)
        out, err = popen.communicate()
        if popen.returncode != 0:
            raise AssertionError(f'process failed with code {popen.returncode}:stderr follows\n{err.decode()}\n')

    def test_caching(self):
        self.check_pycache(0)
        self.run_in_separate_process(1)
        self.check_pycache(3 * 2)
        self.run_in_separate_process(2)
        self.check_pycache(3 * 2)