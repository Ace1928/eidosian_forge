import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def _get_non_admin_token(self):
    non_admin_auth_data = self.build_authentication_request(user_id=self.non_admin_user['id'], password=self.non_admin_user['password'], project_id=self.project['id'])
    return self.get_requested_token(non_admin_auth_data)