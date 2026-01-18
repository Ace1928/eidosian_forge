import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
class TestFlowControl(TestCase):

    def run_test(self, pyfunc, x_operands, y_operands, flags=enable_pyobj_flags):
        cfunc = jit((types.intp, types.intp), **flags)(pyfunc)
        for x, y in itertools.product(x_operands, y_operands):
            pyerr = None
            cerr = None
            try:
                pyres = pyfunc(x, y)
            except Exception as e:
                pyerr = e
            try:
                cres = cfunc(x, y)
            except Exception as e:
                if pyerr is None:
                    raise
                cerr = e
                self.assertEqual(type(pyerr), type(cerr))
            else:
                if pyerr is not None:
                    self.fail('Invalid for pure-python but numba works\n' + pyerr)
                self.assertEqual(pyres, cres)

    def test_for_loop1(self, flags=enable_pyobj_flags):
        self.run_test(for_loop_usecase1, [-10, 0, 10], [0], flags=flags)

    def test_for_loop1_npm(self):
        self.test_for_loop1(flags=no_pyobj_flags)

    def test_for_loop2(self, flags=enable_pyobj_flags):
        self.run_test(for_loop_usecase2, [-10, 0, 10], [-10, 0, 10], flags=flags)

    def test_for_loop2_npm(self):
        self.test_for_loop2(flags=no_pyobj_flags)

    def test_for_loop3(self, flags=enable_pyobj_flags):
        """
        List requires pyobject
        """
        self.run_test(for_loop_usecase3, [1], [2], flags=flags)

    def test_for_loop3_npm(self):
        self.test_for_loop3(flags=no_pyobj_flags)

    def test_for_loop4(self, flags=enable_pyobj_flags):
        self.run_test(for_loop_usecase4, [10], [10], flags=flags)

    def test_for_loop4_npm(self):
        self.test_for_loop4(flags=no_pyobj_flags)

    def test_for_loop5(self, flags=enable_pyobj_flags):
        self.run_test(for_loop_usecase5, [100], [50], flags=flags)

    def test_for_loop5_npm(self):
        self.test_for_loop5(flags=no_pyobj_flags)

    def test_for_loop6(self, flags=enable_pyobj_flags):
        self.run_test(for_loop_usecase6, [100], [50], flags=flags)

    def test_for_loop6_npm(self):
        self.test_for_loop6(flags=no_pyobj_flags)

    def test_for_loop7(self, flags=enable_pyobj_flags):
        self.run_test(for_loop_usecase7, [5], [0], flags=flags)

    def test_for_loop7_npm(self):
        self.test_for_loop7(flags=no_pyobj_flags)

    def test_for_loop8(self, flags=enable_pyobj_flags):
        self.run_test(for_loop_usecase8, [0, 1], [0, 2, 10], flags=flags)

    def test_for_loop8_npm(self):
        self.test_for_loop8(flags=no_pyobj_flags)

    def test_for_loop9(self, flags=enable_pyobj_flags):
        self.run_test(for_loop_usecase9, [0, 1], [0, 2, 10], flags=flags)

    def test_for_loop9_npm(self):
        self.test_for_loop9(flags=no_pyobj_flags)

    def test_for_loop10(self, flags=enable_pyobj_flags):
        self.run_test(for_loop_usecase10, [5], [2, 7], flags=flags)

    def test_for_loop10_npm(self):
        self.test_for_loop10(flags=no_pyobj_flags)

    def test_while_loop1(self, flags=enable_pyobj_flags):
        self.run_test(while_loop_usecase1, [10], [0], flags=flags)

    def test_while_loop1_npm(self):
        self.test_while_loop1(flags=no_pyobj_flags)

    def test_while_loop2(self, flags=enable_pyobj_flags):
        self.run_test(while_loop_usecase2, [10], [0], flags=flags)

    def test_while_loop2_npm(self):
        self.test_while_loop2(flags=no_pyobj_flags)

    def test_while_loop3(self, flags=enable_pyobj_flags):
        self.run_test(while_loop_usecase3, [10], [10], flags=flags)

    def test_while_loop3_npm(self):
        self.test_while_loop3(flags=no_pyobj_flags)

    def test_while_loop4(self, flags=enable_pyobj_flags):
        self.run_test(while_loop_usecase4, [10], [0], flags=flags)

    def test_while_loop4_npm(self):
        self.test_while_loop4(flags=no_pyobj_flags)

    def test_while_loop5(self, flags=enable_pyobj_flags):
        self.run_test(while_loop_usecase5, [0, 5, 10], [0, 5, 10], flags=flags)

    def test_while_loop5_npm(self):
        self.test_while_loop5(flags=no_pyobj_flags)

    def test_ifelse1(self, flags=enable_pyobj_flags):
        self.run_test(ifelse_usecase1, [-1, 0, 1], [-1, 0, 1], flags=flags)

    def test_ifelse1_npm(self):
        self.test_ifelse1(flags=no_pyobj_flags)

    def test_ifelse2(self, flags=enable_pyobj_flags):
        self.run_test(ifelse_usecase2, [-1, 0, 1], [-1, 0, 1], flags=flags)

    def test_ifelse2_npm(self):
        self.test_ifelse2(flags=no_pyobj_flags)

    def test_ifelse3(self, flags=enable_pyobj_flags):
        self.run_test(ifelse_usecase3, [-1, 0, 1], [-1, 0, 1], flags=flags)

    def test_ifelse3_npm(self):
        self.test_ifelse3(flags=no_pyobj_flags)

    def test_ifelse4(self, flags=enable_pyobj_flags):
        self.run_test(ifelse_usecase4, [-1, 0, 1], [-1, 0, 1], flags=flags)

    def test_ifelse4_npm(self):
        self.test_ifelse4(flags=no_pyobj_flags)

    def test_ternary_ifelse1(self, flags=enable_pyobj_flags):
        self.run_test(ternary_ifelse_usecase1, [-1, 0, 1], [-1, 0, 1], flags=flags)

    def test_ternary_ifelse1_npm(self):
        self.test_ternary_ifelse1(flags=no_pyobj_flags)

    def test_double_infinite_loop(self, flags=enable_pyobj_flags):
        self.run_test(double_infinite_loop, [10], [0], flags=flags)

    def test_double_infinite_loop_npm(self):
        self.test_double_infinite_loop(flags=no_pyobj_flags)