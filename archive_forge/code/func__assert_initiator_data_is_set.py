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