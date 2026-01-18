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
def _assert_notify_not_sent(self, resource_id, operation, resource_type, public=True):
    unexpected = {'resource_id': resource_id, 'operation': operation, 'resource_type': resource_type, 'send_notification_called': True, 'public': public}
    for note in self._notifications:
        self.assertNotEqual(unexpected, note)