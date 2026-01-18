import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
class TestTraitEventNotifierEqual(unittest.TestCase):
    """ Test comparing two instances of TraitEventNotifier. """

    def setUp(self):
        push_exception_handler(reraise_exceptions=True)
        self.addCleanup(pop_exception_handler)

    def tearDown(self):
        pass

    def test_equals_use_handler_and_target(self):
        handler1 = mock.Mock()
        handler2 = mock.Mock()
        target1 = mock.Mock()
        target2 = mock.Mock()
        dispatcher = dispatch_here
        notifier1 = create_notifier(handler=handler1, target=target1, dispatcher=dispatcher)
        notifier2 = create_notifier(handler=handler1, target=target1, dispatcher=dispatcher)
        notifier3 = create_notifier(handler=handler1, target=target2, dispatcher=dispatcher)
        notifier4 = create_notifier(handler=handler2, target=target1, dispatcher=dispatcher)
        self.assertTrue(notifier1.equals(notifier2), 'The two notifiers should consider each other as equal.')
        self.assertTrue(notifier2.equals(notifier1), 'The two notifiers should consider each other as equal.')
        self.assertFalse(notifier3.equals(notifier1), 'Expected the notifiers to be different because targets are not identical.')
        self.assertFalse(notifier4.equals(notifier1), 'Expected the notifiers to be different because the handlers do not compare equally.')

    def test_equality_check_with_instance_methods(self):
        instance = DummyObservable()
        target = mock.Mock()
        notifier1 = create_notifier(handler=instance.handler, target=target)
        notifier2 = create_notifier(handler=instance.handler, target=target)
        self.assertTrue(notifier1.equals(notifier2))
        self.assertTrue(notifier2.equals(notifier1))

    def test_equals_compared_to_different_type(self):
        notifier = create_notifier()
        self.assertFalse(notifier.equals(float))

    def test_not_equal_if_dispatcher_different(self):
        handler = mock.Mock()
        target = mock.Mock()
        dispatcher1 = mock.Mock()
        dispatcher2 = mock.Mock()
        notifier1 = create_notifier(handler=handler, target=target, dispatcher=dispatcher1)
        notifier2 = create_notifier(handler=handler, target=target, dispatcher=dispatcher2)
        self.assertFalse(notifier1.equals(notifier2), 'Expected the notifiers to be different because the dispatchers do not compare equally.')
        self.assertFalse(notifier2.equals(notifier1), 'Expected the notifiers to be different because the dispatchers do not compare equally.')