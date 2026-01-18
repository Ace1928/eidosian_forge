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
def _test_role_assignment(self, url, role, project=None, domain=None, user=None, group=None):
    self.put(url)
    action = '%s.%s' % (CREATED_OPERATION, self.ROLE_ASSIGNMENT)
    event_type = '%s.%s.%s' % (notifications.SERVICE, self.ROLE_ASSIGNMENT, CREATED_OPERATION)
    self._assert_last_note(action, self.user_id, event_type)
    self._assert_event(role, project, domain, user, group)
    self.delete(url)
    action = '%s.%s' % (DELETED_OPERATION, self.ROLE_ASSIGNMENT)
    event_type = '%s.%s.%s' % (notifications.SERVICE, self.ROLE_ASSIGNMENT, DELETED_OPERATION)
    self._assert_last_note(action, self.user_id, event_type)
    self._assert_event(role, project, domain, user, None)