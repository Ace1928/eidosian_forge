import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
class TestDataFlow(TestCase):

    def test_assignments(self, flags=force_pyobj_jit_opt):
        pyfunc = assignments
        cfunc = jit((types.int32,), **flags)(pyfunc)
        for x in [-1, 0, 1]:
            self.assertPreciseEqual(pyfunc(x), cfunc(x))

    def test_assignments2(self, flags=force_pyobj_jit_opt):
        pyfunc = assignments2
        cfunc = jit((types.int32,), **flags)(pyfunc)
        for x in [-1, 0, 1]:
            self.assertPreciseEqual(pyfunc(x), cfunc(x))
        if flags is force_pyobj_jit_opt:
            cfunc('a')

    def run_propagate_func(self, func, args):
        self.assertPreciseEqual(func(*args), func.py_func(*args))

    def test_var_propagate1(self):
        cfunc = njit((types.intp, types.intp))(var_propagate1)
        self.run_propagate_func(cfunc, (2, 3))
        self.run_propagate_func(cfunc, (3, 2))

    def test_var_propagate2(self):
        cfunc = njit((types.intp, types.intp))(var_propagate2)
        self.run_propagate_func(cfunc, (2, 3))
        self.run_propagate_func(cfunc, (3, 2))

    def test_var_propagate3(self):
        cfunc = njit((types.intp, types.intp))(var_propagate3)
        self.run_propagate_func(cfunc, (2, 3))
        self.run_propagate_func(cfunc, (3, 2))
        self.run_propagate_func(cfunc, (2, 0))
        self.run_propagate_func(cfunc, (-1, 0))
        self.run_propagate_func(cfunc, (0, 2))
        self.run_propagate_func(cfunc, (0, -1))

    def test_var_propagate4(self):
        cfunc = njit((types.intp, types.intp))(var_propagate4)
        self.run_propagate_func(cfunc, (1, 1))
        self.run_propagate_func(cfunc, (1, 0))
        self.run_propagate_func(cfunc, (1, -1))
        self.run_propagate_func(cfunc, (0, 1))
        self.run_propagate_func(cfunc, (0, 0))
        self.run_propagate_func(cfunc, (0, -1))
        self.run_propagate_func(cfunc, (-1, 1))
        self.run_propagate_func(cfunc, (-1, 0))
        self.run_propagate_func(cfunc, (-1, -1))

    def test_chained_compare(self, flags=force_pyobj_jit_opt):
        pyfunc = chained_compare
        cfunc = jit((types.int32,), **flags)(pyfunc)
        for x in [0, 1, 2, 3, 4]:
            self.assertPreciseEqual(pyfunc(x), cfunc(x))

    def test_chained_compare_npm(self):
        self.test_chained_compare(no_pyobj_jit_opt)

    def test_stack_effect_error(self, flags=force_pyobj_jit_opt):
        pyfunc = stack_effect_error
        cfunc = jit((types.int32,), **flags)(pyfunc)
        for x in (0, 1, 2, 3):
            self.assertPreciseEqual(pyfunc(x), cfunc(x))

    def test_stack_effect_error_npm(self):
        self.test_stack_effect_error(no_pyobj_jit_opt)

    def test_var_swapping(self, flags=force_pyobj_jit_opt):
        pyfunc = var_swapping
        cfunc = jit((types.int32,) * 5, **flags)(pyfunc)
        args = tuple(range(0, 10, 2))
        self.assertPreciseEqual(pyfunc(*args), cfunc(*args))

    def test_var_swapping_npm(self):
        self.test_var_swapping(no_pyobj_jit_opt)

    def test_for_break(self, flags=force_pyobj_jit_opt):
        pyfunc = for_break
        cfunc = jit((types.intp, types.intp), **flags)(pyfunc)
        for n, x in [(4, 2), (4, 6)]:
            self.assertPreciseEqual(pyfunc(n, x), cfunc(n, x))

    def test_for_break_npm(self):
        self.test_for_break(no_pyobj_jit_opt)