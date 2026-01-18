import faulthandler
import itertools
import multiprocessing
import os
import random
import re
import subprocess
import sys
import textwrap
import threading
import unittest
import numpy as np
from numba import jit, vectorize, guvectorize, set_num_threads
from numba.tests.support import (temp_directory, override_config, TestCase, tag,
import queue as t_queue
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
from numba.core import config
class ThreadLayerTestHelper(TestCase):
    """
    Helper class for running an isolated piece of code based on a template
    """
    _here = '%r' % os.path.dirname(__file__)
    template = 'if 1:\n    import sys\n    sys.path.insert(0, "%(here)r")\n    import multiprocessing\n    import numpy as np\n    from numba import njit\n    import numba\n    try:\n        import threading_backend_usecases\n    except ImportError as e:\n        print("DEBUG:", sys.path)\n        raise e\n    import os\n\n    sigterm_handler = threading_backend_usecases.sigterm_handler\n    busy_func = threading_backend_usecases.busy_func\n\n    def the_test():\n        %%s\n\n    if __name__ == "__main__":\n        the_test()\n    ' % {'here': _here}

    def run_cmd(self, cmdline, env=None):
        if env is None:
            env = os.environ.copy()
            env['NUMBA_THREADING_LAYER'] = str('omp')
        popen = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        timeout = threading.Timer(_TEST_TIMEOUT, popen.kill)
        try:
            timeout.start()
            out, err = popen.communicate()
            if popen.returncode != 0:
                raise AssertionError('process failed with code %s: stderr follows\n%s\n' % (popen.returncode, err.decode()))
        finally:
            timeout.cancel()
        return (out.decode(), err.decode())