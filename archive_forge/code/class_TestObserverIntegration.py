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
class TestObserverIntegration(unittest.TestCase):
    """ Test the public facing observe function."""

    def setUp(self):
        push_exception_handler(reraise_exceptions=True)
        self.addCleanup(pop_exception_handler)

    def test_observe_with_expression(self):
        foo = ClassWithNumber()
        handler = mock.Mock()
        observe(object=foo, expression=trait('number'), handler=handler)
        foo.number += 1
        self.assertEqual(handler.call_count, 1)
        handler.reset_mock()
        observe(object=foo, expression=trait('number'), handler=handler, remove=True)
        foo.number += 1
        self.assertEqual(handler.call_count, 0)

    def test_observe_different_dispatcher(self):
        self.dispatch_records = []

        def dispatcher(handler, event):
            self.dispatch_records.append((handler, event))
        foo = ClassWithNumber()
        handler = mock.Mock()
        observe(object=foo, expression=trait('number'), handler=handler, dispatcher=dispatcher)
        foo.number += 1
        self.assertEqual(len(self.dispatch_records), 1)

    def test_observe_different_target(self):
        parent1 = ClassWithInstance()
        parent2 = ClassWithInstance()
        instance = ClassWithNumber()
        parent1.instance = instance
        parent2.instance = instance
        handler = mock.Mock()
        observe(object=parent1, expression=trait('instance').trait('number'), handler=handler)
        observe(object=parent2, expression=trait('instance').trait('number'), handler=handler)
        instance.number += 1
        self.assertEqual(handler.call_count, 2)

    def test_observe_with_any_callables_accepting_one_argument(self):

        def handler_with_one_pos_arg(arg, *, optional=None):
            pass
        callables = [repr, lambda e: False, handler_with_one_pos_arg]
        for callable_ in callables:
            with self.subTest(callable=callable_):
                instance = ClassWithNumber()
                instance.observe(callable_, 'number')
                instance.number += 1