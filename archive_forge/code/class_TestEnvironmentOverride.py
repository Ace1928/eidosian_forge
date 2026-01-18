import os
import platform
import re
import textwrap
import warnings
import numpy as np
from numba.tests.support import (TestCase, override_config, override_env_config,
from numba import jit, njit
from numba.core import types, compiler, utils
from numba.core.errors import NumbaPerformanceWarning
from numba import prange
from numba.experimental import jitclass
import unittest
class TestEnvironmentOverride(FunctionDebugTestBase):
    """
    Test that environment variables are reloaded by Numba when modified.
    """
    _numba_parallel_test_ = False

    def test_debug(self):
        out = self.compile_simple_nopython()
        self.assertFalse(out)
        with override_env_config('NUMBA_DEBUG', '1'):
            out = self.compile_simple_nopython()
            self.check_debug_output(out, ['ir', 'typeinfer', 'llvm', 'func_opt_llvm', 'optimized_llvm', 'assembly'])
        out = self.compile_simple_nopython()
        self.assertFalse(out)