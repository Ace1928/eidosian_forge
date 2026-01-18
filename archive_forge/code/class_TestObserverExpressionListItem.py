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
class TestObserverExpressionListItem(unittest.TestCase):
    """ Test ObserverExpression.list_items """

    def test_list_items(self):
        expr = expression.list_items()
        expected = [create_graph(ListItemObserver(notify=True, optional=False))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_list_items_notify_false(self):
        expr = expression.list_items(notify=False)
        expected = [create_graph(ListItemObserver(notify=False, optional=False))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_list_items_optional_true(self):
        expr = expression.list_items(optional=True)
        expected = [create_graph(ListItemObserver(notify=True, optional=True))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_list_items_method_notify(self):
        expr = expression.list_items().list_items(notify=False)
        expected = [create_graph(ListItemObserver(notify=True, optional=False), ListItemObserver(notify=False, optional=False))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_list_items_method_optional(self):
        expr = expression.list_items().list_items(optional=True)
        expected = [create_graph(ListItemObserver(notify=True, optional=False), ListItemObserver(notify=True, optional=True))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_call_signatures(self):
        top_level = expression.list_items
        method = expression.ObserverExpression().list_items
        self.assertEqual(inspect.signature(top_level), inspect.signature(method))