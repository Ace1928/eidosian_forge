import contextlib
import gc
import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Str, WeakRef
from traits.testing.unittest_tools import UnittestTools
class TestWeakRef(UnittestTools, unittest.TestCase):
    """ Test cases for weakref (WeakRef) traits. """

    def test_set_and_get(self):
        eggs = Eggs(name='platypus')
        spam = Spam()
        self.assertIsNone(spam.eggs)
        spam.eggs = eggs
        self.assertIs(spam.eggs, eggs)
        del eggs
        self.assertIsNone(spam.eggs)

    def test_target_freed_notification(self):
        eggs = Eggs(name='duck')
        spam = Spam(eggs=eggs)
        with self.assertTraitChanges(spam, 'eggs'):
            del eggs

    def test_weakref_trait_doesnt_leak_cycles(self):
        eggs = Eggs(name='ostrich')
        with restore_gc_state():
            gc.disable()
            gc.collect()
            spam = Spam(eggs=eggs)
            del spam
            self.assertEqual(gc.collect(), 0)