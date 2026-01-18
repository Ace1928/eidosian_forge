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
class TestObserverExpressionEqualityAndHashing(unittest.TestCase):
    """ Test ObserverExpression.__eq__ and ObserverExpression.__hash__. """

    def test_trait_equality(self):
        expr1 = create_expression(1)
        expr2 = create_expression(1)
        self.assertEqual(expr1, expr2)
        self.assertEqual(hash(expr1), hash(expr2))

    def test_join_equality_with_then(self):
        expr1 = create_expression(1)
        expr2 = create_expression(2)
        combined1 = expression.join(expr1, expr2)
        combined2 = expr1.then(expr2)
        self.assertEqual(combined1, combined2)
        self.assertEqual(hash(combined1), hash(combined2))

    def test_equality_of_parallel_expressions(self):
        expr1 = create_expression(1) | create_expression(2)
        expr2 = create_expression(1) | create_expression(2)
        self.assertEqual(expr1, expr2)
        self.assertEqual(hash(expr1), hash(expr2))

    def test_equality_different_type(self):
        expr = create_expression(1)
        self.assertNotEqual(expr, '1')