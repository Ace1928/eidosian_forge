import collections
import weakref
import gc
import operator
from itertools import takewhile
import unittest
from numba import njit, jit
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.untyped_passes import PreserveIR
from numba.core.typed_passes import IRLegalization
from numba.core import types, ir
from numba.tests.support import TestCase, override_config, SerialMixin
class TestObjLifetime(TestCase):
    """
    Test lifetime of Python objects inside jit-compiled functions.
    """

    def compile(self, pyfunc):
        cfunc = jit((types.pyobject,), forceobj=True, looplift=False)(pyfunc)
        return cfunc

    def compile_and_record(self, pyfunc, raises=None):
        rec = RefRecorder()
        cfunc = self.compile(pyfunc)
        if raises is not None:
            with self.assertRaises(raises):
                cfunc(rec)
        else:
            cfunc(rec)
        return rec

    def assertRecordOrder(self, rec, expected):
        """
        Check that the *expected* markers occur in that order in *rec*'s
        recorded events.
        """
        actual = []
        recorded = rec.recorded
        remaining = list(expected)
        for d in recorded:
            if d in remaining:
                actual.append(d)
                remaining.remove(d)
        self.assertEqual(actual, expected, 'the full list of recorded events is: %r' % (recorded,))

    def test_simple1(self):
        rec = self.compile_and_record(simple_usecase1)
        self.assertFalse(rec.alive)
        self.assertRecordOrder(rec, ['a', 'b', '--1--'])
        self.assertRecordOrder(rec, ['a', 'c', '--1--'])
        self.assertRecordOrder(rec, ['--1--', 'b + c', '--2--'])

    def test_simple2(self):
        rec = self.compile_and_record(simple_usecase2)
        self.assertFalse(rec.alive)
        self.assertRecordOrder(rec, ['b', '--1--', 'a'])

    def test_looping1(self):
        rec = self.compile_and_record(looping_usecase1)
        self.assertFalse(rec.alive)
        self.assertRecordOrder(rec, ['a', 'b', '--loop exit--', 'c'])
        self.assertRecordOrder(rec, ['iter(a)#1', '--loop bottom--', 'iter(a)#2', '--loop bottom--', 'iter(a)#3', '--loop bottom--', 'iter(a)', '--loop exit--'])

    def test_looping2(self):
        rec = self.compile_and_record(looping_usecase2)
        self.assertFalse(rec.alive)
        self.assertRecordOrder(rec, ['a', '--outer loop top--'])
        self.assertRecordOrder(rec, ['iter(a)', '--outer loop else--', '--outer loop exit--'])
        self.assertRecordOrder(rec, ['iter(b)', '--inner loop exit #1--', 'iter(b)', '--inner loop exit #2--', 'iter(b)', '--inner loop exit #3--'])
        self.assertRecordOrder(rec, ['iter(a)#1', '--inner loop entry #1--', 'iter(a)#2', '--inner loop entry #2--', 'iter(a)#3', '--inner loop entry #3--'])
        self.assertRecordOrder(rec, ['iter(a)#1 + iter(a)#1', '--outer loop bottom #1--'])

    def exercise_generator(self, genfunc):
        cfunc = self.compile(genfunc)
        rec = RefRecorder()
        with self.assertRefCount(rec):
            gen = cfunc(rec)
            next(gen)
            self.assertTrue(rec.alive)
            list(gen)
            self.assertFalse(rec.alive)
        rec = RefRecorder()
        with self.assertRefCount(rec):
            gen = cfunc(rec)
            del gen
            gc.collect()
            self.assertFalse(rec.alive)
        rec = RefRecorder()
        with self.assertRefCount(rec):
            gen = cfunc(rec)
            next(gen)
            self.assertTrue(rec.alive)
            del gen
            gc.collect()
            self.assertFalse(rec.alive)

    def test_generator1(self):
        self.exercise_generator(generator_usecase1)

    def test_generator2(self):
        self.exercise_generator(generator_usecase2)

    def test_del_before_definition(self):
        rec = self.compile_and_record(del_before_definition)
        self.assertEqual(rec.recorded, ['0', '1', '2'])

    def test_raising1(self):
        with self.assertRefCount(do_raise):
            rec = self.compile_and_record(raising_usecase1, raises=MyError)
            self.assertFalse(rec.alive)

    def test_raising2(self):
        with self.assertRefCount(do_raise):
            rec = self.compile_and_record(raising_usecase2, raises=MyError)
            self.assertFalse(rec.alive)

    def test_raising3(self):
        with self.assertRefCount(MyError):
            rec = self.compile_and_record(raising_usecase3, raises=MyError)
            self.assertFalse(rec.alive)

    def test_inf_loop_multiple_back_edge(self):
        cfunc = self.compile(inf_loop_multiple_back_edge)
        rec = RefRecorder()
        iterator = iter(cfunc(rec))
        next(iterator)
        self.assertEqual(rec.alive, [])
        next(iterator)
        self.assertEqual(rec.alive, [])
        next(iterator)
        self.assertEqual(rec.alive, [])
        self.assertEqual(rec.recorded, ['yield', 'p', 'bra', 'yield', 'p', 'bra', 'yield'])