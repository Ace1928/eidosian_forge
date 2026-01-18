from unittest import mock
from oslotest import base
import testtools
from neutron_lib.callbacks import events
from neutron_lib.callbacks import priority_group
from neutron_lib.callbacks import registry
from neutron_lib.callbacks import resources
from neutron_lib import fixture
class CallBacksManagerTestCase(base.BaseTestCase):

    def test_decorated_inst_method_receives(self):
        i1 = ObjectWithDecoratedCallback()
        event_payload = events.EventPayload(mock.ANY)
        registry.publish(resources.PORT, events.BEFORE_CREATE, self, payload=event_payload)
        self.assertEqual(0, i1.counter)
        registry.publish(resources.PORT, events.AFTER_CREATE, self, payload=event_payload)
        self.assertEqual(1, i1.counter)
        registry.publish(resources.PORT, events.AFTER_UPDATE, self, payload=event_payload)
        self.assertEqual(2, i1.counter)
        registry.publish(resources.NETWORK, events.AFTER_UPDATE, self, payload=event_payload)
        self.assertEqual(2, i1.counter)
        registry.publish(resources.NETWORK, events.AFTER_DELETE, self, payload=event_payload)
        self.assertEqual(3, i1.counter)
        i2 = ObjectWithDecoratedCallback()
        self.assertEqual(0, i2.counter)
        registry.publish(resources.NETWORK, events.AFTER_DELETE, self, payload=event_payload)
        self.assertEqual(4, i1.counter)
        self.assertEqual(1, i2.counter)

    def test_object_inheriting_others_no_double_subscribe(self):
        with mock.patch.object(registry, 'subscribe') as sub:
            callback = AnotherObjectWithDecoratedCallback()
            priority_call = [mock.call(callback.callback2, resources.NETWORK, events.AFTER_DELETE, PRI_CALLBACK)]
            self.assertEqual(4, len(sub.mock_calls))
            sub.assert_has_calls(priority_call)

    def test_new_inheritance_not_broken(self):
        self.assertTrue(AnotherObjectWithDecoratedCallback().new_called)

    def test_object_new_not_broken(self):
        CallbackClassWithParameters('dummy')

    def test_no_strings_in_events_arg(self):
        with testtools.ExpectedException(AssertionError):
            registry.receives(resources.PORT, events.AFTER_CREATE)