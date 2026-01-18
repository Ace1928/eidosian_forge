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
def _test_notification_operation_with_basic_format(self, notify_function, operation):
    self.config_fixture.config(notification_format='basic')
    exp_resource_id = uuid.uuid4().hex
    callback = register_callback(operation)
    notify_function(EXP_RESOURCE_TYPE, exp_resource_id)
    callback.assert_called_once_with('identity', EXP_RESOURCE_TYPE, operation, {'resource_info': exp_resource_id})