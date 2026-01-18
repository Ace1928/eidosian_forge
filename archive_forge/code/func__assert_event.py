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
def _assert_event(self, role_id, project=None, domain=None, user=None, group=None, inherit=False):
    """Assert that the CADF event is valid.

        In the case of role assignments, the event will have extra data,
        specifically, the role, target, actor, and if the role is inherited.

        An example event, as a dictionary is seen below:
            {
                'typeURI': 'http://schemas.dmtf.org/cloud/audit/1.0/event',
                'initiator': {
                    'typeURI': 'service/security/account/user',
                    'host': {'address': 'localhost'},
                    'id': 'openstack:0a90d95d-582c-4efb-9cbc-e2ca7ca9c341',
                    'username': u'admin'
                },
                'target': {
                    'typeURI': 'service/security/account/user',
                    'id': 'openstack:d48ea485-ef70-4f65-8d2b-01aa9d7ec12d'
                },
                'observer': {
                    'typeURI': 'service/security',
                    'id': 'openstack:d51dd870-d929-4aba-8d75-dcd7555a0c95'
                },
                'eventType': 'activity',
                'eventTime': '2014-08-21T21:04:56.204536+0000',
                'role': u'0e6b990380154a2599ce6b6e91548a68',
                'domain': u'24bdcff1aab8474895dbaac509793de1',
                'inherited_to_projects': False,
                'group': u'c1e22dc67cbd469ea0e33bf428fe597a',
                'action': 'created.role_assignment',
                'outcome': 'success',
                'id': 'openstack:782689dd-f428-4f13-99c7-5c70f94a5ac1'
            }
        """
    note = self._notifications[-1]
    event = note['event']
    if project:
        self.assertEqual(project, event.project)
    if domain:
        self.assertEqual(domain, event.domain)
    if group:
        self.assertEqual(group, event.group)
    elif user:
        self.assertEqual(user, event.user)
    self.assertEqual(role_id, event.role)
    self.assertEqual(inherit, event.inherited_to_projects)