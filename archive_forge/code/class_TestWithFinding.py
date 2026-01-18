import copy
import warnings
import numpy as np
import numba
from numba.core.transforms import find_setupwiths, with_lifting
from numba.core.withcontexts import bypass_context, call_context, objmode_context
from numba.core.bytecode import FunctionIdentity, ByteCode
from numba.core.interpreter import Interpreter
from numba.core import errors
from numba.core.registry import cpu_target
from numba.core.compiler import compile_ir, DEFAULT_FLAGS
from numba import njit, typeof, objmode, types
from numba.core.extending import overload
from numba.tests.support import (MemoryLeak, TestCase, captured_stdout,
from numba.core.utils import PYVERSION
from numba.experimental import jitclass
import unittest
class TestWithFinding(TestCase):

    def check_num_of_with(self, func, expect_count):
        the_ir = get_func_ir(func)
        ct = len(find_setupwiths(the_ir)[0])
        self.assertEqual(ct, expect_count)

    def test_lift1(self):
        self.check_num_of_with(lift1, expect_count=1)

    def test_lift2(self):
        self.check_num_of_with(lift2, expect_count=2)

    def test_lift3(self):
        self.check_num_of_with(lift3, expect_count=1)

    def test_lift4(self):
        self.check_num_of_with(lift4, expect_count=2)

    def test_lift5(self):
        self.check_num_of_with(lift5, expect_count=0)