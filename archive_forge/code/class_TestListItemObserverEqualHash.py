import unittest
from unittest import mock
from traits.api import HasTraits, Instance, Int, List
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._testing import (
from traits.trait_list_object import TraitList, TraitListObject
class TestListItemObserverEqualHash(unittest.TestCase):

    def test_not_equal_notify(self):
        observer1 = ListItemObserver(notify=False, optional=False)
        observer2 = ListItemObserver(notify=True, optional=False)
        self.assertNotEqual(observer1, observer2)

    def test_not_equal_optional(self):
        observer1 = ListItemObserver(notify=True, optional=True)
        observer2 = ListItemObserver(notify=True, optional=False)
        self.assertNotEqual(observer1, observer2)

    def test_not_equal_different_type(self):
        observer1 = ListItemObserver(notify=False, optional=False)
        imposter = mock.Mock()
        imposter.notify = False
        imposter.optional = False
        self.assertNotEqual(observer1, imposter)

    def test_equal_observers(self):
        observer1 = ListItemObserver(notify=False, optional=False)
        observer2 = ListItemObserver(notify=False, optional=False)
        self.assertEqual(observer1, observer2)
        self.assertEqual(hash(observer1), hash(observer2))

    def test_slots(self):
        observer = ListItemObserver(notify=True, optional=False)
        with self.assertRaises(AttributeError):
            observer.__dict__
        with self.assertRaises(AttributeError):
            observer.__weakref__

    def test_eval_repr_roundtrip(self):
        observer = ListItemObserver(notify=True, optional=False)
        self.assertEqual(eval(repr(observer)), observer)