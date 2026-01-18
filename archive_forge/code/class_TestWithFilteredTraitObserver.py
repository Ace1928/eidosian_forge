import unittest
from unittest import mock
from traits.api import HasTraits, Int
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._metadata_filter import MetadataFilter
from traits.observation._testing import (
class TestWithFilteredTraitObserver(unittest.TestCase):
    """ Test MetadataFilter with FilteredTraitObserver and HasTraits. """

    def test_filter_metadata(self):

        class Person(HasTraits):
            n_jobs = Int(status='public')
            n_children = Int()
        observer = FilteredTraitObserver(filter=MetadataFilter(metadata_name='status'), notify=True)
        person = Person()
        handler = mock.Mock()
        call_add_or_remove_notifiers(object=person, graph=create_graph(observer), handler=handler)
        person.n_jobs += 1
        self.assertEqual(handler.call_count, 1)
        handler.reset_mock()
        person.n_children += 1
        self.assertEqual(handler.call_count, 0)