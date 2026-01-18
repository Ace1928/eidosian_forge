from contextlib import contextmanager
import numpy as np
from numba import cuda
from numba.cuda.testing import (unittest, skip_on_cudasim,
from numba.tests.support import captured_stderr
from numba.core import config
class TestDeferCleanupAvail(CUDATestCase):

    def test_context_manager(self):
        with cuda.defer_cleanup():
            pass