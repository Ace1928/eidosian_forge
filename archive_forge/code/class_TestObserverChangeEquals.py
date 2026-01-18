import unittest
from unittest import mock
import weakref
from traits.api import HasTraits, Instance, Int
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation.exceptions import NotifierNotFound
class TestObserverChangeEquals(unittest.TestCase):
    """ Test ObserverChangeNotifier.equals """

    def test_notifier_equals(self):
        observer_handler = mock.Mock()
        handler = mock.Mock()
        graph = mock.Mock()
        target = mock.Mock()
        notifier1 = create_notifier(observer_handler=observer_handler, handler=handler, graph=graph, target=target, dispatcher=dispatch_here)
        notifier2 = create_notifier(observer_handler=observer_handler, handler=handler, graph=graph, target=target, dispatcher=dispatch_here)
        self.assertTrue(notifier1.equals(notifier2), 'Expected notifier1 to see notifier2 as equals.')
        self.assertTrue(notifier2.equals(notifier1), 'Expected notifier2 to see notifier1 as equals.')

    def test_notifier_observer_handler_not_equal(self):
        handler = mock.Mock()
        graph = mock.Mock()
        target = mock.Mock()
        notifier1 = create_notifier(observer_handler=mock.Mock(), handler=handler, graph=graph, target=target, dispatcher=dispatch_here)
        notifier2 = create_notifier(observer_handler=mock.Mock(), handler=handler, graph=graph, target=target, dispatcher=dispatch_here)
        self.assertFalse(notifier1.equals(notifier2), 'Expected notifier1 to see notifier2 as different.')
        self.assertFalse(notifier2.equals(notifier1), 'Expected notifier2 to see notifier1 as different.')

    def test_notifier_handler_not_equal(self):
        observer_handler = mock.Mock()
        graph = mock.Mock()
        target = mock.Mock()
        notifier1 = create_notifier(observer_handler=observer_handler, handler=mock.Mock(), graph=graph, target=target, dispatcher=dispatch_here)
        notifier2 = create_notifier(observer_handler=observer_handler, handler=mock.Mock(), graph=graph, target=target, dispatcher=dispatch_here)
        self.assertFalse(notifier1.equals(notifier2), 'Expected notifier1 to see notifier2 as different.')
        self.assertFalse(notifier2.equals(notifier1), 'Expected notifier2 to see notifier1 as different.')

    def test_notifier_graph_not_equal(self):
        observer_handler = mock.Mock()
        handler = mock.Mock()
        target = mock.Mock()
        notifier1 = create_notifier(observer_handler=observer_handler, handler=handler, graph=mock.Mock(), target=target, dispatcher=dispatch_here)
        notifier2 = create_notifier(observer_handler=observer_handler, handler=handler, graph=mock.Mock(), target=target, dispatcher=dispatch_here)
        self.assertFalse(notifier1.equals(notifier2), 'Expected notifier1 to see notifier2 as different.')
        self.assertFalse(notifier2.equals(notifier1), 'Expected notifier2 to see notifier1 as different.')

    def test_notifier_target_not_equals(self):
        observer_handler = mock.Mock()
        handler = mock.Mock()
        graph = mock.Mock()
        target1 = mock.Mock()
        target2 = mock.Mock()
        notifier1 = create_notifier(observer_handler=observer_handler, handler=handler, graph=graph, target=target1, dispatcher=dispatch_here)
        notifier2 = create_notifier(observer_handler=observer_handler, handler=handler, graph=graph, target=target2, dispatcher=dispatch_here)
        self.assertFalse(notifier1.equals(notifier2), 'Expected notifier1 to see notifier2 as different.')
        self.assertFalse(notifier2.equals(notifier1), 'Expected notifier2 to see notifier1 as different.')

    def test_notifier_dispatcher_not_equals(self):
        observer_handler = mock.Mock()
        handler = mock.Mock()
        graph = mock.Mock()
        target = mock.Mock()
        dispatcher1 = mock.Mock()
        dispatcher2 = mock.Mock()
        notifier1 = create_notifier(observer_handler=observer_handler, handler=handler, graph=graph, target=target, dispatcher=dispatcher1)
        notifier2 = create_notifier(observer_handler=observer_handler, handler=handler, graph=graph, target=target, dispatcher=dispatcher2)
        self.assertFalse(notifier1.equals(notifier2), 'Expected notifier1 to see notifier2 as different.')
        self.assertFalse(notifier2.equals(notifier1), 'Expected notifier2 to see notifier1 as different.')

    def test_notifier_equals_graphs_compared_for_equality(self):
        graph1 = tuple([1, 2, 3])
        graph2 = tuple([1, 2, 3])
        observer_handler = mock.Mock()
        handler = mock.Mock()
        target = mock.Mock()
        notifier1 = create_notifier(observer_handler=observer_handler, handler=handler, graph=graph1, target=target, dispatcher=dispatch_here)
        notifier2 = create_notifier(observer_handler=observer_handler, handler=handler, graph=graph2, target=target, dispatcher=dispatch_here)
        self.assertTrue(notifier1.equals(notifier2), 'Expected notifier1 to see notifier2 as equals.')
        self.assertTrue(notifier2.equals(notifier1), 'Expected notifier2 to see notifier1 as equals.')

    def test_notifier_equals_with_different_type(self):
        notifier = create_notifier()
        self.assertFalse(notifier.equals(str))

    def test_notifier_instance_method_handler_equal(self):
        observer_handler = mock.Mock()
        graph = mock.Mock()
        target = mock.Mock()
        instance = DummyClass()
        notifier1 = create_notifier(observer_handler=observer_handler, handler=instance.dummy_method, graph=graph, target=target, dispatcher=dispatch_here)
        notifier2 = create_notifier(observer_handler=observer_handler, handler=instance.dummy_method, graph=graph, target=target, dispatcher=dispatch_here)
        self.assertTrue(notifier1.equals(notifier2), 'Expected notifier1 to see notifier2 as equals.')
        self.assertTrue(notifier2.equals(notifier1), 'Expected notifier2 to see notifier1 as equals.')