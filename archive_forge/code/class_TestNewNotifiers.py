import threading
import unittest
from unittest import mock
from traits.api import Float, HasTraits, List
from traits.testing.unittest_tools import UnittestTools
class TestNewNotifiers(UnittestTools, unittest.TestCase):
    """ Tests for dynamic notifiers with `dispatch='new'`. """

    def test_notification_on_separate_thread(self):
        receiver = Receiver()

        def on_foo_notifications(obj, name, old, new):
            thread_id = threading.current_thread().ident
            event = (thread_id, obj, name, old, new)
            receiver.notifications.append(event)
        obj = Foo()
        obj.on_trait_change(on_foo_notifications, 'foo', dispatch='new')
        with RememberThreads() as remember_threads:
            patcher = mock.patch('traits.trait_notifiers.Thread', new=remember_threads)
            with patcher:
                obj.foo = 3
            self.assertEventuallyTrue(receiver, 'notifications_items', Receiver.notified, timeout=SAFETY_TIMEOUT)
        notifications = receiver.notifications
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0][1:], (obj, 'foo', 0, 3))
        this_thread_id = threading.current_thread().ident
        self.assertNotEqual(this_thread_id, notifications[0][0])