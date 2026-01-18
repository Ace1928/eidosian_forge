from io import StringIO
import numpy as np
from numba.core import types
from numba.core.compiler import compile_extra, Flags
from numba.tests.support import TestCase, tag, MemoryLeakMixin
import unittest
class TestLoopLifting(MemoryLeakMixin, TestCase):

    def try_lift(self, pyfunc, argtypes):
        from numba.core.registry import cpu_target
        cres = compile_extra(cpu_target.typing_context, cpu_target.target_context, pyfunc, argtypes, return_type=None, flags=looplift_flags, locals={})
        self.assertEqual(len(cres.lifted), 1)
        return cres

    def assert_lifted_native(self, cres):
        jitloop = cres.lifted[0]
        [loopcres] = jitloop.overloads.values()
        self.assertTrue(loopcres.fndesc.native)

    def check_lift_ok(self, pyfunc, argtypes, args):
        """
        Check that pyfunc can loop-lift even in nopython mode.
        """
        cres = self.try_lift(pyfunc, argtypes)
        expected = pyfunc(*args)
        got = cres.entry_point(*args)
        self.assert_lifted_native(cres)
        self.assertPreciseEqual(expected, got)

    def check_lift_generator_ok(self, pyfunc, argtypes, args):
        """
        Check that pyfunc (a generator function) can loop-lift even in
        nopython mode.
        """
        cres = self.try_lift(pyfunc, argtypes)
        expected = list(pyfunc(*args))
        got = list(cres.entry_point(*args))
        self.assert_lifted_native(cres)
        self.assertPreciseEqual(expected, got)

    def check_no_lift(self, pyfunc, argtypes, args):
        """
        Check that pyfunc can't loop-lift.
        """
        cres = compile_isolated(pyfunc, argtypes, flags=looplift_flags)
        self.assertFalse(cres.lifted)
        expected = pyfunc(*args)
        got = cres.entry_point(*args)
        self.assertPreciseEqual(expected, got)

    def check_no_lift_generator(self, pyfunc, argtypes, args):
        """
        Check that pyfunc (a generator function) can't loop-lift.
        """
        cres = compile_isolated(pyfunc, argtypes, flags=looplift_flags)
        self.assertFalse(cres.lifted)
        expected = list(pyfunc(*args))
        got = list(cres.entry_point(*args))
        self.assertPreciseEqual(expected, got)

    def test_lift1(self):
        self.check_lift_ok(lift1, (types.intp,), (123,))

    def test_lift2(self):
        self.check_lift_ok(lift2, (types.intp,), (123,))

    def test_lift3(self):
        self.check_lift_ok(lift3, (types.intp,), (123,))

    def test_lift4(self):
        self.check_lift_ok(lift4, (types.intp,), (123,))

    def test_lift5(self):
        self.check_lift_ok(lift5, (types.intp,), (123,))

    def test_lift_issue2561(self):
        self.check_lift_ok(lift_issue2561, (), ())

    def test_lift_gen1(self):
        self.check_lift_generator_ok(lift_gen1, (types.intp,), (123,))

    def test_reject1(self):
        self.check_no_lift(reject1, (types.intp,), (123,))

    def test_reject_gen1(self):
        self.check_no_lift_generator(reject_gen1, (types.intp,), (123,))

    def test_reject_gen2(self):
        self.check_no_lift_generator(reject_gen2, (types.intp,), (123,))