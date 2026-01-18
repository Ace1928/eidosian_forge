import numpy as np
from numba.core.utils import PYVERSION
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
from numba.tests.support import (override_config, captured_stderr,
from numba import cuda, float64
import unittest
def _check_dump_ir(self, out):
    self.assertIn('--IR DUMP: simple_cuda--', out)
    self.assertIn('const(float, 1.5)', out)