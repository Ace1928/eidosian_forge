import os
import sys
import subprocess
import threading
from numba import cuda
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
from numba.tests.support import captured_stdout
@skip_under_cuda_memcheck('Hangs cuda-memcheck')
class TestCUDAFindLibs(CUDATestCase):

    def run_cmd(self, cmdline, env):
        popen = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        timeout = threading.Timer(5 * 60.0, popen.kill)
        try:
            timeout.start()
            out, err = popen.communicate()
            return (out.decode(), err.decode())
        finally:
            timeout.cancel()
        return (None, None)

    def run_test_in_separate_process(self, envvar, envvar_value):
        env_copy = os.environ.copy()
        env_copy[envvar] = str(envvar_value)
        code = "if 1:\n            from numba import cuda\n            @cuda.jit('(int64,)')\n            def kernel(x):\n                pass\n            kernel(1,)\n            "
        cmdline = [sys.executable, '-c', code]
        return self.run_cmd(cmdline, env_copy)

    @skip_on_cudasim('Simulator does not hit device library search code path')
    @unittest.skipIf(not sys.platform.startswith('linux'), 'linux only')
    def test_cuda_find_lib_errors(self):
        """
        This tests that the find_libs works as expected in the case of an
        environment variable being used to set the path.
        """
        locs = ['lib', 'lib64']
        looking_for = None
        for l in locs:
            looking_for = os.path.join(os.path.sep, l)
            if os.path.exists(looking_for):
                break
        if looking_for is not None:
            out, err = self.run_test_in_separate_process('NUMBA_CUDA_DRIVER', looking_for)
            self.assertTrue(out is not None)
            self.assertTrue(err is not None)