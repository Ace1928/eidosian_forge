import gc
from io import StringIO
import numpy as np
from numba import njit, vectorize
from numba import typeof
from numba.core import utils, types, typing, ir, compiler, cpu, cgutils
from numba.core.compiler import Compiler, Flags
from numba.core.registry import cpu_target
from numba.tests.support import (MemoryLeakMixin, TestCase, temp_directory,
from numba.extending import (
import operator
import textwrap
import unittest
def _assert_total_rewrite(self, control_ir, test_ir, trivial=False):
    """
        Given two dictionaries of Numba IR blocks, check to make sure the
        control IR has no array expressions, while the test IR
        contains one and only one.
        """
    self.assertEqual(len(control_ir), len(test_ir))
    control_block = control_ir[0].body
    test_block = test_ir[0].body
    self._assert_array_exprs(control_block, 0)
    self._assert_array_exprs(test_block, 1)
    if not trivial:
        self.assertGreater(len(control_block), len(test_block))