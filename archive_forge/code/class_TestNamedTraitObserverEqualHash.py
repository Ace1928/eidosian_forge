import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
class TestNamedTraitObserverEqualHash(unittest.TestCase):
    """ Unit tests on the NamedTraitObserver __eq__ and __hash__ methods."""

    def test_not_equal_notify(self):
        observer1 = NamedTraitObserver(name='foo', notify=True, optional=True)
        observer2 = NamedTraitObserver(name='foo', notify=False, optional=True)
        self.assertNotEqual(observer1, observer2)

    def test_not_equal_name(self):
        observer1 = NamedTraitObserver(name='foo', notify=True, optional=True)
        observer2 = NamedTraitObserver(name='bar', notify=True, optional=True)
        self.assertNotEqual(observer1, observer2)

    def test_not_equal_optional(self):
        observer1 = NamedTraitObserver(name='foo', notify=True, optional=False)
        observer2 = NamedTraitObserver(name='foo', notify=True, optional=True)
        self.assertNotEqual(observer1, observer2)

    def test_equal_observers(self):
        observer1 = NamedTraitObserver(name='foo', notify=True, optional=True)
        observer2 = NamedTraitObserver(name='foo', notify=True, optional=True)
        self.assertEqual(observer1, observer2)
        self.assertEqual(hash(observer1), hash(observer2))

    def test_not_equal_type(self):
        observer = NamedTraitObserver(name='foo', notify=True, optional=True)
        imposter = mock.Mock()
        imposter.name = 'foo'
        imposter.notify = True
        imposter.optional = True
        self.assertNotEqual(observer, imposter)

    def test_slots(self):
        observer = NamedTraitObserver(name='foo', notify=True, optional=True)
        with self.assertRaises(AttributeError):
            observer.__dict__
        with self.assertRaises(AttributeError):
            observer.__weakref__

    def test_eval_repr_roundtrip(self):
        observer = NamedTraitObserver(name='foo', notify=True, optional=True)
        self.assertEqual(eval(repr(observer)), observer)