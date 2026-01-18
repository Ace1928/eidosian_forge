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
class TestLiftCall(BaseTestWithLifting):

    def check_same_semantic(self, func):
        """Ensure same semantic with non-jitted code
        """
        jitted = njit(func)
        with captured_stdout() as got:
            jitted()
        with captured_stdout() as expect:
            func()
        self.assertEqual(got.getvalue(), expect.getvalue())

    def test_liftcall1(self):
        self.check_extracted_with(liftcall1, expect_count=1, expected_stdout='A 1\nB 2\n')
        self.check_same_semantic(liftcall1)

    def test_liftcall2(self):
        self.check_extracted_with(liftcall2, expect_count=2, expected_stdout='A 1\nB 2\nC 12\n')
        self.check_same_semantic(liftcall2)

    def test_liftcall3(self):
        self.check_extracted_with(liftcall3, expect_count=2, expected_stdout='A 1\nB 2\nC 47\n')
        self.check_same_semantic(liftcall3)

    def test_liftcall4(self):
        accept = (errors.TypingError, errors.NumbaRuntimeError, errors.NumbaValueError, errors.CompilerError)
        with self.assertRaises(accept) as raises:
            njit(liftcall4)()
        msg = 'compiler re-entrant to the same function signature'
        self.assertIn(msg, str(raises.exception))

    @expected_failure_py311
    @expected_failure_py312
    def test_liftcall5(self):
        self.check_extracted_with(liftcall5, expect_count=1, expected_stdout='0\n1\n2\n3\n4\n5\nA\n')
        self.check_same_semantic(liftcall5)