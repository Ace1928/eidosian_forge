import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
def _create_test_roles(self):
    ref = unit.new_role_ref()
    role = PROVIDERS.role_api.create_role(ref['id'], ref)
    self.prior_role_id = role['id']
    ref = unit.new_role_ref()
    role = PROVIDERS.role_api.create_role(ref['id'], ref)
    self.implied_role_id = role['id']