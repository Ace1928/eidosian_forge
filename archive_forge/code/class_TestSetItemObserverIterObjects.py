import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._set_item_observer import SetItemObserver
from traits.observation._testing import (
from traits.trait_set_object import TraitSet
from traits.trait_types import Set
class TestSetItemObserverIterObjects(unittest.TestCase):
    """ Test SetItemObserver.iter_objects """

    def test_iter_objects_from_set(self):
        instance = ClassWithSet()
        instance.values = set([1, 2, 3])
        observer = create_observer()
        actual = list(observer.iter_objects(instance.values))
        self.assertCountEqual(actual, [1, 2, 3])

    def test_iter_observables_custom_trait_set(self):
        custom_trait_set = CustomTraitSet([1, 2, 3])
        observer = create_observer()
        actual = list(observer.iter_objects(custom_trait_set))
        self.assertCountEqual(actual, [1, 2, 3])

    def test_iter_objects_sanity_check(self):
        observer = create_observer(optional=False)
        with self.assertRaises(ValueError) as exception_context:
            list(observer.iter_objects(None))
        self.assertIn('Expected a TraitSet to be observed, got', str(exception_context.exception))

    def test_iter_objects_optional(self):
        observer = create_observer(optional=True)
        actual = list(observer.iter_objects(None))
        self.assertEqual(actual, [])