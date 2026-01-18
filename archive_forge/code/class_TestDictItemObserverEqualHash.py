import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._testing import (
from traits.trait_dict_object import TraitDict
from traits.trait_types import Dict, Str
class TestDictItemObserverEqualHash(unittest.TestCase):
    """ Test DictItemObserver __eq__, __hash__. """

    def test_not_equal_notify(self):
        observer1 = DictItemObserver(notify=False, optional=False)
        observer2 = DictItemObserver(notify=True, optional=False)
        self.assertNotEqual(observer1, observer2)

    def test_not_equal_optional(self):
        observer1 = DictItemObserver(notify=True, optional=True)
        observer2 = DictItemObserver(notify=True, optional=False)
        self.assertNotEqual(observer1, observer2)

    def test_not_equal_different_type(self):
        observer1 = DictItemObserver(notify=False, optional=False)
        imposter = mock.Mock()
        imposter.notify = False
        imposter.optional = False
        self.assertNotEqual(observer1, imposter)

    def test_equal_observers(self):
        observer1 = DictItemObserver(notify=False, optional=False)
        observer2 = DictItemObserver(notify=False, optional=False)
        self.assertEqual(observer1, observer2)
        self.assertEqual(hash(observer1), hash(observer2))

    def test_slots(self):
        observer = DictItemObserver(notify=True, optional=False)
        with self.assertRaises(AttributeError):
            observer.__dict__
        with self.assertRaises(AttributeError):
            observer.__weakref__

    def test_eval_repr_roundtrip(self):
        observer = DictItemObserver(notify=True, optional=False)
        self.assertEqual(eval(repr(observer)), observer)