import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._testing import (
from traits.observation._trait_added_observer import (
from traits.trait_types import Str
class TestRestrictedNamedTraitObserverEqualityHash(unittest.TestCase):
    """ Test _RestrictedNamedTraitObserver.__eq__ and __hash__ """

    def test_equality_name_and_observer(self):
        wrapped_observer = DummyObserver()
        observer1 = _RestrictedNamedTraitObserver(name='name', wrapped_observer=wrapped_observer)
        observer2 = _RestrictedNamedTraitObserver(name='name', wrapped_observer=wrapped_observer)
        self.assertEqual(observer1, observer2)
        self.assertEqual(hash(observer1), hash(observer2))

    def test_not_equal_name(self):
        wrapped_observer = DummyObserver()
        observer1 = _RestrictedNamedTraitObserver(name='name', wrapped_observer=wrapped_observer)
        observer2 = _RestrictedNamedTraitObserver(name='other', wrapped_observer=wrapped_observer)
        self.assertNotEqual(observer1, observer2)

    def test_not_equal_observer(self):
        observer1 = _RestrictedNamedTraitObserver(name='name', wrapped_observer=DummyObserver())
        observer2 = _RestrictedNamedTraitObserver(name='name', wrapped_observer=DummyObserver())
        self.assertNotEqual(observer1, observer2)