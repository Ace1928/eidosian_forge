import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.trait_base import Undefined, Uninitialized
from traits.trait_types import Float, Instance, Int, List
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._testing import (
class TestFilteredTraitObserverIterObjects(unittest.TestCase):
    """ Test FilteredTraitObserver.iter_objects """

    def test_iter_objects(self):
        instance = DummyParent()
        instance.instance = Dummy()
        self.assertIsNone(instance.instance2)
        observer = create_observer(filter=lambda name, trait: type(trait.trait_type) is Instance)
        actual = list(observer.iter_objects(instance))
        self.assertEqual(actual, [instance.instance])
        instance.instance2 = Dummy()
        actual = list(observer.iter_objects(instance))
        self.assertCountEqual(actual, [instance.instance, instance.instance2])