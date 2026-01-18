import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_log import log
import oslo_messaging
from pycadf import cadftaxonomy
from pycadf import cadftype
from pycadf import eventfactory
from pycadf import resource as cadfresource
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import test_v3
class BaseNotificationTest(test_v3.RestfulTestCase):

    def setUp(self):
        super(BaseNotificationTest, self).setUp()
        self._notifications = []
        self._audits = []

        def fake_notify(operation, resource_type, resource_id, initiator=None, actor_dict=None, public=True):
            note = {'resource_id': resource_id, 'operation': operation, 'resource_type': resource_type, 'initiator': initiator, 'send_notification_called': True, 'public': public}
            if actor_dict:
                note['actor_id'] = actor_dict.get('id')
                note['actor_type'] = actor_dict.get('type')
                note['actor_operation'] = actor_dict.get('actor_operation')
            self._notifications.append(note)
        self.useFixture(fixtures.MockPatchObject(notifications, '_send_notification', fake_notify))

        def fake_audit(action, initiator, outcome, target, event_type, reason=None, **kwargs):
            service_security = cadftaxonomy.SERVICE_SECURITY
            event = eventfactory.EventFactory().new_event(eventType=cadftype.EVENTTYPE_ACTIVITY, outcome=outcome, action=action, initiator=initiator, target=target, reason=reason, observer=cadfresource.Resource(typeURI=service_security))
            for key, value in kwargs.items():
                setattr(event, key, value)
            payload = event.as_dict()
            audit = {'payload': payload, 'event_type': event_type, 'send_notification_called': True}
            self._audits.append(audit)
        self.useFixture(fixtures.MockPatchObject(notifications, '_send_audit_notification', fake_audit))

    def _assert_last_note(self, resource_id, operation, resource_type, actor_id=None, actor_type=None, actor_operation=None):
        if CONF.notification_format != 'basic':
            return
        self.assertGreater(len(self._notifications), 0)
        note = self._notifications[-1]
        self.assertEqual(operation, note['operation'])
        self.assertEqual(resource_id, note['resource_id'])
        self.assertEqual(resource_type, note['resource_type'])
        self.assertTrue(note['send_notification_called'])
        if actor_id:
            self.assertEqual(actor_id, note['actor_id'])
            self.assertEqual(actor_type, note['actor_type'])
            self.assertEqual(actor_operation, note['actor_operation'])

    def _assert_last_audit(self, resource_id, operation, resource_type, target_uri, reason=None):
        if CONF.notification_format != 'cadf':
            return
        self.assertGreater(len(self._audits), 0)
        audit = self._audits[-1]
        payload = audit['payload']
        if 'resource_info' in payload:
            self.assertEqual(resource_id, payload['resource_info'])
        action = '.'.join(filter(None, [operation, resource_type]))
        self.assertEqual(action, payload['action'])
        self.assertEqual(target_uri, payload['target']['typeURI'])
        if resource_id:
            self.assertEqual(resource_id, payload['target']['id'])
        event_type = '.'.join(filter(None, ['identity', resource_type, operation]))
        self.assertEqual(event_type, audit['event_type'])
        if reason:
            self.assertEqual(reason['reasonCode'], payload['reason']['reasonCode'])
            self.assertEqual(reason['reasonType'], payload['reason']['reasonType'])
        self.assertTrue(audit['send_notification_called'])

    def _assert_initiator_data_is_set(self, operation, resource_type, typeURI):
        self.assertGreater(len(self._audits), 0)
        audit = self._audits[-1]
        payload = audit['payload']
        self.assertEqual(self.user_id, payload['initiator']['id'])
        self.assertEqual(self.project_id, payload['initiator']['project_id'])
        self.assertEqual(typeURI, payload['target']['typeURI'])
        self.assertIn('request_id', payload['initiator'])
        action = '%s.%s' % (operation, resource_type)
        self.assertEqual(action, payload['action'])

    def _assert_notify_not_sent(self, resource_id, operation, resource_type, public=True):
        unexpected = {'resource_id': resource_id, 'operation': operation, 'resource_type': resource_type, 'send_notification_called': True, 'public': public}
        for note in self._notifications:
            self.assertNotEqual(unexpected, note)

    def _assert_notify_sent(self, resource_id, operation, resource_type, public=True):
        expected = {'resource_id': resource_id, 'operation': operation, 'resource_type': resource_type, 'send_notification_called': True, 'public': public}
        for note in self._notifications:
            if all((note.get(k) == v for k, v in expected.items())):
                break
        else:
            self.fail('Notification not sent.')