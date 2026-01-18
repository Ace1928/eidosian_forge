import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import unittest
from _pydevd_frame_eval.vendored.bytecode import ConcreteBytecode, Bytecode, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode.tests import get_code
class CodeTests(unittest.TestCase):
    """Check that bytecode.from_code(code).to_code() returns code."""

    def check(self, source, function=False):
        ref_code = get_code(source, function=function)
        code = ConcreteBytecode.from_code(ref_code).to_code()
        self.assertEqual(code, ref_code)
        code = Bytecode.from_code(ref_code).to_code()
        self.assertEqual(code, ref_code)
        bytecode = Bytecode.from_code(ref_code)
        blocks = ControlFlowGraph.from_bytecode(bytecode)
        code = blocks.to_bytecode().to_code()
        self.assertEqual(code, ref_code)

    def test_loop(self):
        self.check('\n            for x in range(1, 10):\n                x += 1\n                if x == 3:\n                    continue\n                x -= 1\n                if x > 7:\n                    break\n                x = 0\n            print(x)\n        ')

    def test_varargs(self):
        self.check('\n            def func(a, b, *varargs):\n                pass\n        ', function=True)

    def test_kwargs(self):
        self.check('\n            def func(a, b, **kwargs):\n                pass\n        ', function=True)

    def test_kwonlyargs(self):
        self.check('\n            def func(*, arg, arg2):\n                pass\n        ', function=True)

    def test_generator_func(self):
        self.check('\n            def func(arg, arg2):\n                yield\n        ', function=True)

    def test_async_func(self):
        self.check('\n            async def func(arg, arg2):\n                pass\n        ', function=True)