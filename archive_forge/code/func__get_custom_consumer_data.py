import calendar
from unittest import mock
from barbicanclient import exceptions as barbican_exceptions
from keystoneauth1 import identity
from keystoneauth1 import service_token
from oslo_context import context
from oslo_utils import timeutils
from oslo_utils import uuidutils
from castellan.common import exception
from castellan.common.objects import symmetric_key as sym_key
from castellan.key_manager import barbican_key_manager
from castellan.tests.unit.key_manager import test_key_manager
def _get_custom_consumer_data(self, service='storage', resource_type='volume', resource_id=uuidutils.generate_uuid()):
    return {'service': service, 'resource_type': resource_type, 'resource_id': resource_id}