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
class TestObserverExpressionComposition(unittest.TestCase):
    """ Test composition of ObserverExpression with generic observers."""

    def test_new_with_branches(self):
        observer = 1
        expr = create_expression(observer)
        expected = [create_graph(observer)]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_or_operator(self):
        observer1 = 1
        observer2 = 2
        expr1 = create_expression(observer1)
        expr2 = create_expression(observer2)
        expr = expr1 | expr2
        expected = [create_graph(observer1), create_graph(observer2)]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_or_maintain_order(self):
        observer1 = 1
        observer2 = 2
        expr1 = create_expression(observer1)
        expr2 = create_expression(observer2)
        combined1 = expr1 | expr2
        combined2 = expr2 | expr1
        self.assertEqual(combined1._as_graphs(), combined2._as_graphs()[::-1])

    def test_then_operator(self):
        observer1 = 1
        observer2 = 2
        expr1 = create_expression(observer1)
        expr2 = create_expression(observer2)
        expr = expr1.then(expr2)
        expected = [create_graph(observer1, observer2)]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_chained_then_or(self):
        observer1 = 1
        observer2 = 2
        observer3 = 3
        observer4 = 4
        expr1 = create_expression(observer1)
        expr2 = create_expression(observer2)
        expr3 = create_expression(observer3)
        expr4 = create_expression(observer4)
        expr = expr1.then(expr2) | expr3.then(expr4)
        expected = [create_graph(observer1, observer2), create_graph(observer3, observer4)]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_or_then_chained(self):
        observer1 = 1
        observer2 = 2
        observer3 = 3
        observer4 = 4
        expr1 = create_expression(observer1)
        expr2 = create_expression(observer2)
        expr3 = create_expression(observer3)
        expr4 = create_expression(observer4)
        expr = (expr1 | expr2).then(expr3 | expr4)
        expected = [ObserverGraph(node=observer1, children=[create_graph(observer3), create_graph(observer4)]), ObserverGraph(node=observer2, children=[create_graph(observer3), create_graph(observer4)])]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_join_expressions(self):
        observer1 = 1
        observer2 = 2
        expr1 = create_expression(observer1)
        expr2 = create_expression(observer2)
        expr = expression.join(expr1, expr2)
        expected = [create_graph(observer1, observer2)]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)