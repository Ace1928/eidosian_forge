import threading
import time
import unittest
from traits import trait_notifiers
from traits.api import Callable, Float, HasTraits, on_trait_change
class BaseTestUINotifiers(object):
    """ Tests for dynamic notifiers with `dispatch='ui'`.
    """

    def setUp(self):
        self.notifications = []

    def flush_event_loop(self):
        """ Post and process the Qt events. """
        qt4_app.sendPostedEvents()
        qt4_app.processEvents()

    def on_foo_notifications(self, obj, name, old, new):
        thread_id = threading.current_thread().ident
        event = (thread_id, (obj, name, old, new))
        self.notifications.append(event)

    @unittest.skipIf(not QT_FOUND, 'Qt event loop not found, UI dispatch not possible.')
    def test_notification_from_main_thread(self):
        obj = self.obj_factory()
        obj.foo = 3
        self.flush_event_loop()
        notifications = self.notifications
        self.assertEqual(len(notifications), 1)
        thread_id, event = notifications[0]
        self.assertEqual(event, (obj, 'foo', 0, 3))
        ui_thread = trait_notifiers.ui_thread
        self.assertEqual(thread_id, ui_thread)

    @unittest.skipIf(not QT_FOUND, 'Qt event loop not found, UI dispatch not possible.')
    def test_notification_from_separate_thread(self):
        obj = self.obj_factory()

        def set_foo_to_3(obj):
            obj.foo = 3
        threading.Thread(target=set_foo_to_3, args=(obj,)).start()
        time.sleep(0.1)
        self.flush_event_loop()
        notifications = self.notifications
        self.assertEqual(len(notifications), 1)
        thread_id, event = notifications[0]
        self.assertEqual(event, (obj, 'foo', 0, 3))
        ui_thread = trait_notifiers.ui_thread
        self.assertEqual(thread_id, ui_thread)