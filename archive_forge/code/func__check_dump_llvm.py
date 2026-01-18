import numpy as np
from numba.core.utils import PYVERSION
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
from numba.tests.support import (override_config, captured_stderr,
from numba import cuda, float64
import unittest
def _check_dump_llvm(self, out):
    self.assertIn('--LLVM DUMP', out)
    self.assertIn('!"kernel", i32 1', out)