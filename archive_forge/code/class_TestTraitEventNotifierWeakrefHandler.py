import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
class TestTraitEventNotifierWeakrefHandler(unittest.TestCase):
    """ Test weakref handling for handler in TraitEventNotifier."""

    def setUp(self):
        push_exception_handler(reraise_exceptions=True)
        self.addCleanup(pop_exception_handler)

    def tearDown(self):
        pass

    def test_method_as_handler_does_not_prevent_garbage_collect(self):
        dummy = DummyObservable()
        dummy.internal_object = DummyObservable()
        dummy_ref = weakref.ref(dummy)
        notifier = create_notifier(handler=dummy.handler)
        notifier.add_to(dummy.internal_object)
        del dummy
        self.assertIsNone(dummy_ref())

    def test_callable_disabled_if_handler_deleted(self):
        dummy = DummyObservable()
        dummy.internal_object = DummyObservable()
        event_factory = mock.Mock()
        notifier = create_notifier(handler=dummy.handler, event_factory=event_factory)
        notifier.add_to(dummy.internal_object)
        notifier(a=1, b=2)
        self.assertEqual(event_factory.call_count, 1)
        event_factory.reset_mock()
        del dummy
        notifier(a=1, b=2)
        event_factory.assert_not_called()

    def test_reference_held_when_dispatching(self):
        dummy = DummyObservable()

        def event_factory(*args, **kwargs):
            nonlocal dummy
            del dummy
        notifier = create_notifier(handler=dummy.handler, event_factory=event_factory)
        notifier.add_to(dummy)
        notifier(a=1, b=2)