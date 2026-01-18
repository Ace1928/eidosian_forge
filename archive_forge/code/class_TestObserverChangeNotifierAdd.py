import unittest
from unittest import mock
import weakref
from traits.api import HasTraits, Instance, Int
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation.exceptions import NotifierNotFound
class TestObserverChangeNotifierAdd(unittest.TestCase):
    """ Test ObserverChangeNotifier.add_to """

    def test_add_notifier(self):
        instance = DummyClass()
        notifier = create_notifier()
        notifier.add_to(instance)
        self.assertEqual(instance.notifiers, [notifier])

    def test_add_to_ignore_same_notifier(self):
        handler = mock.Mock()
        observer_handler = mock.Mock()
        graph = mock.Mock()
        target = mock.Mock()
        notifier1 = create_notifier(observer_handler=observer_handler, graph=graph, handler=handler, target=target)
        notifier2 = create_notifier(observer_handler=observer_handler, graph=graph, handler=handler, target=target)
        instance = DummyClass()
        notifier1.add_to(instance)
        notifier2.add_to(instance)
        self.assertEqual(instance.notifiers, [notifier1, notifier2])