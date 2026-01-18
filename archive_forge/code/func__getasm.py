from numba.tests.support import override_config
from numba.cuda.testing import skip_on_cudasim
from numba import cuda
from numba.core import types
from numba.cuda.testing import CUDATestCase
import itertools
import re
import unittest
def _getasm(self, fn, sig):
    fn.compile(sig)
    return fn.inspect_asm(sig)