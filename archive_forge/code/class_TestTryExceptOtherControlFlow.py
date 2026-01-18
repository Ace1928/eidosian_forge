import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
class TestTryExceptOtherControlFlow(TestCase):

    def test_yield(self):

        @njit
        def udt(n, x):
            for i in range(n):
                try:
                    if i == x:
                        raise ValueError
                    yield i
                except Exception:
                    return
        self.assertEqual(list(udt(10, 5)), list(range(5)))
        self.assertEqual(list(udt(10, 10)), list(range(10)))

    @expected_failure_py311
    @expected_failure_py312
    def test_objmode(self):

        @njit
        def udt():
            try:
                with objmode():
                    print(object())
            except Exception:
                return
        with self.assertRaises(CompilerError) as raises:
            udt()
        msg = 'unsupported control flow: with-context contains branches (i.e. break/return/raise) that can leave the block '
        self.assertIn(msg, str(raises.exception))

    @expected_failure_py311
    @expected_failure_py312
    def test_objmode_output_type(self):

        def bar(x):
            return np.asarray(list(reversed(x.tolist())))

        @njit
        def test_objmode():
            x = np.arange(5)
            y = np.zeros_like(x)
            try:
                with objmode(y='intp[:]'):
                    y += bar(x)
            except Exception:
                pass
            return y
        with self.assertRaises(CompilerError) as raises:
            test_objmode()
        msg = 'unsupported control flow: with-context contains branches (i.e. break/return/raise) that can leave the block '
        self.assertIn(msg, str(raises.exception))

    @unittest.skipIf(PYVERSION < (3, 9), 'Python 3.9+ only')
    def test_reraise_opcode_unreachable(self):

        def pyfunc():
            try:
                raise Exception
            except Exception:
                raise ValueError('ERROR')
        for inst in dis.get_instructions(pyfunc):
            if inst.opname == 'RERAISE':
                break
        else:
            self.fail('expected RERAISE opcode not found')
        func_ir = ir_utils.get_ir_of_code({}, pyfunc.__code__)
        found = False
        for lbl, blk in func_ir.blocks.items():
            for stmt in blk.find_insts(ir.StaticRaise):
                msg = 'Unreachable condition reached (op code RERAISE executed)'
                if stmt.exc_args and msg in stmt.exc_args[0]:
                    found = True
        if not found:
            self.fail('expected RERAISE unreachable message not found')