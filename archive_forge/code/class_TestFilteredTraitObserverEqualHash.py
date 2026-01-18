import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.trait_base import Undefined, Uninitialized
from traits.trait_types import Float, Instance, Int, List
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._testing import (
class TestFilteredTraitObserverEqualHash(unittest.TestCase):
    """ Tests for FilteredTraitObserver __eq__ and __hash__ methods.
    """

    def test_not_equal_filter(self):
        observer1 = FilteredTraitObserver(notify=True, filter=DummyFilter(return_value=True))
        observer2 = FilteredTraitObserver(notify=True, filter=DummyFilter(return_value=False))
        self.assertNotEqual(observer1, observer2)

    def test_not_equal_notify(self):
        filter_func = mock.Mock()
        observer1 = FilteredTraitObserver(notify=False, filter=filter_func)
        observer2 = FilteredTraitObserver(notify=True, filter=filter_func)
        self.assertNotEqual(observer1, observer2)

    def test_equal_filter_notify(self):
        observer1 = FilteredTraitObserver(notify=True, filter=DummyFilter(return_value=True))
        observer2 = FilteredTraitObserver(notify=True, filter=DummyFilter(return_value=True))
        self.assertEqual(observer1, observer2)
        self.assertEqual(hash(observer1), hash(observer2))

    def test_not_equal_type(self):
        filter_func = mock.Mock()
        observer1 = FilteredTraitObserver(notify=True, filter=filter_func)
        imposter = mock.Mock()
        imposter.notify = True
        imposter.filter = filter_func
        self.assertNotEqual(observer1, imposter)

    def test_slots(self):
        observer = FilteredTraitObserver(notify=True, filter=DummyFilter(return_value=True))
        with self.assertRaises(AttributeError):
            observer.__dict__
        with self.assertRaises(AttributeError):
            observer.__weakref__

    def test_eval_repr_roundtrip(self):
        observer = FilteredTraitObserver(notify=True, filter=DummyFilter(return_value=True))
        self.assertEqual(eval(repr(observer)), observer)