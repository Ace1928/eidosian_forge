import unittest
from unittest import mock
from traits.api import HasTraits, Instance, Int, List
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._testing import (
from traits.trait_list_object import TraitList, TraitListObject
class TestListItemObserverIterObjects(unittest.TestCase):
    """ Test ListItemObserver.iter_objects """

    def test_trait_list_iter_objects(self):
        instance = ClassWithList()
        item1 = mock.Mock()
        item2 = mock.Mock()
        instance.values = [item1, item2]
        observer = ListItemObserver(notify=True, optional=False)
        actual = list(observer.iter_objects(instance.values))
        self.assertEqual(actual, [item1, item2])

    def test_trait_list_iter_object_accept_custom_trait_list(self):
        instance = ClassWithList()
        instance.custom_trait_list = CustomTraitList([1, 2, 3])
        observer = ListItemObserver(notify=True, optional=False)
        actual = list(observer.iter_objects(instance.custom_trait_list))
        self.assertEqual(actual, [1, 2, 3])

    def test_trait_list_iter_objects_complain_not_list(self):
        observer = ListItemObserver(notify=True, optional=False)
        with self.assertRaises(ValueError) as exception_cm:
            next(observer.iter_objects(set([1])))
        self.assertIn('Expected a TraitList to be observed', str(exception_cm.exception))

    def test_trait_list_iter_objects_ignore_if_optional_and_not_list(self):
        observer = ListItemObserver(notify=True, optional=True)
        actual = list(observer.iter_objects(set([1])))
        self.assertEqual(actual, [])