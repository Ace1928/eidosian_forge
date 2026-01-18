import unittest
from unittest import mock
from traits.api import HasTraits, Int
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._metadata_filter import MetadataFilter
from traits.observation._testing import (
class TestMetadataFilter(unittest.TestCase):
    """ Test the behaviour of the MetadataFilter """

    def test_metadata_defined_vs_undefined(self):
        metadata_filter = MetadataFilter(metadata_name='name')
        self.assertTrue(metadata_filter('name', Int(name=True).as_ctrait()), 'Expected the filter to return true')
        self.assertFalse(metadata_filter('name', Int().as_ctrait()), 'Expected the filter to return false')

    def test_metadata_defined_as_none_is_same_as_undefined(self):
        metadata_filter = MetadataFilter(metadata_name='name')
        self.assertFalse(metadata_filter('name', Int(name=None).as_ctrait()), 'Expected the filter to return false')

    def test_filter_equality(self):
        filter1 = MetadataFilter(metadata_name='name')
        filter2 = MetadataFilter(metadata_name='name')
        self.assertEqual(filter1, filter2)
        self.assertEqual(hash(filter1), hash(filter2))

    def test_filter_not_equal_name_different(self):
        filter1 = MetadataFilter(metadata_name='number')
        filter2 = MetadataFilter(metadata_name='name')
        self.assertNotEqual(filter1, filter2)

    def test_filter_not_equal_different_type(self):
        filter1 = MetadataFilter(metadata_name='name')
        imposter = mock.Mock()
        imposter.metadata_name = 'name'
        self.assertNotEqual(imposter, filter1)

    def test_slots(self):
        filter = MetadataFilter(metadata_name='name')
        with self.assertRaises(AttributeError):
            filter.__dict__
        with self.assertRaises(AttributeError):
            filter.__weakref__

    def test_repr_value(self):
        metadata_filter = MetadataFilter(metadata_name='name')
        actual = repr(metadata_filter)
        self.assertEqual(actual, "MetadataFilter(metadata_name='name')")

    def test_eval_repr_roundtrip(self):
        metadata_filter = MetadataFilter(metadata_name='name')
        self.assertEqual(eval(repr(metadata_filter)), metadata_filter)