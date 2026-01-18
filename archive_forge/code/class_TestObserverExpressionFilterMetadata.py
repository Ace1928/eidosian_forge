import inspect
import unittest
from traits.observation import expression
from traits.observation._anytrait_filter import anytrait_filter
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._metadata_filter import MetadataFilter
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._set_item_observer import SetItemObserver
from traits.observation._observer_graph import ObserverGraph
class TestObserverExpressionFilterMetadata(unittest.TestCase):
    """ Test ObserverExpression.metadata """

    def test_metadata_notify_true(self):
        expr = expression.metadata('butterfly')
        expected = [create_graph(FilteredTraitObserver(filter=MetadataFilter(metadata_name='butterfly'), notify=True))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_metadata_notify_false(self):
        expr = expression.metadata('butterfly', notify=False)
        expected = [create_graph(FilteredTraitObserver(filter=MetadataFilter(metadata_name='butterfly'), notify=False))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_metadata_method_notify_true(self):
        expr = expression.metadata('bee').metadata('ant')
        expected = [create_graph(FilteredTraitObserver(filter=MetadataFilter(metadata_name='bee'), notify=True), FilteredTraitObserver(filter=MetadataFilter(metadata_name='ant'), notify=True))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_metadata_method_notify_false(self):
        expr = expression.metadata('bee').metadata('ant', notify=False)
        expected = [create_graph(FilteredTraitObserver(filter=MetadataFilter(metadata_name='bee'), notify=True), FilteredTraitObserver(filter=MetadataFilter(metadata_name='ant'), notify=False))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_call_signatures(self):
        top_level = expression.metadata
        method = expression.ObserverExpression().metadata
        self.assertEqual(inspect.signature(top_level), inspect.signature(method))