import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.trait_types import Instance, Int
from traits.observation.api import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation.expression import compile_expr, trait
from traits.observation.observe import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
class TestApplyObservers(unittest.TestCase):
    """ Test the public-facing apply_observers function."""

    def setUp(self):
        push_exception_handler(reraise_exceptions=True)
        self.addCleanup(pop_exception_handler)

    def test_apply_observers_with_expression(self):
        foo = ClassWithNumber()
        handler = mock.Mock()
        graphs = compile_expr(trait('number'))
        apply_observers(object=foo, graphs=graphs, handler=handler, dispatcher=dispatch_same)
        foo.number += 1
        self.assertEqual(handler.call_count, 1)
        handler.reset_mock()
        apply_observers(object=foo, graphs=graphs, handler=handler, dispatcher=dispatch_same, remove=True)
        foo.number += 1
        self.assertEqual(handler.call_count, 0)

    def test_apply_observers_different_dispatcher(self):
        self.dispatch_records = []

        def dispatcher(handler, event):
            self.dispatch_records.append((handler, event))
        foo = ClassWithNumber()
        handler = mock.Mock()
        apply_observers(object=foo, graphs=compile_expr(trait('number')), handler=handler, dispatcher=dispatcher)
        foo.number += 1
        self.assertEqual(len(self.dispatch_records), 1)

    def test_apply_observers_different_target(self):
        parent1 = ClassWithInstance()
        parent2 = ClassWithInstance()
        graphs = compile_expr(trait('instance').trait('number'))
        instance = ClassWithNumber()
        parent1.instance = instance
        parent2.instance = instance
        handler = mock.Mock()
        apply_observers(object=parent1, graphs=graphs, handler=handler, dispatcher=dispatch_same)
        apply_observers(object=parent2, graphs=graphs, handler=handler, dispatcher=dispatch_same)
        instance.number += 1
        self.assertEqual(handler.call_count, 2)