from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def _create_role(self, domain_id=None):
    new_role = unit.new_role_ref(domain_id=domain_id)
    return PROVIDERS.role_api.create_role(new_role['id'], new_role)