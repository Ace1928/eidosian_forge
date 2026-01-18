import multiprocessing
import platform
import threading
import pickle
import weakref
from itertools import chain
from io import StringIO
import numpy as np
from numba import njit, jit, typeof, vectorize
from numba.core import types, errors
from numba import _dispatcher
from numba.tests.support import TestCase, captured_stdout
from numba.np.numpy_support import as_dtype
from numba.core.dispatcher import Dispatcher
from numba.extending import overload
from numba.tests.support import needs_lapack, SerialMixin
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
import unittest
def _check_cfg_display(self, cfg, wrapper=''):
    if wrapper:
        wrapper = '{}{}'.format(len(wrapper), wrapper)
    module_name = __name__.split('.', 1)[0]
    module_len = len(module_name)
    prefix = '^digraph "CFG for \\\'_ZN{}{}{}'.format(wrapper, module_len, module_name)
    self.assertRegex(str(cfg), prefix)
    self.assertTrue(callable(cfg.display))