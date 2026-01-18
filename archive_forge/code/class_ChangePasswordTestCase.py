import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_db import exception as oslo_db_exception
from oslo_log import log
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import base as identity_base
from keystone.identity.backends import resource_options as options
from keystone.identity.backends import sql_model as model
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
class ChangePasswordTestCase(test_v3.RestfulTestCase):

    def setUp(self):
        super(ChangePasswordTestCase, self).setUp()
        self.user_ref = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
        self.token = self.get_request_token(self.user_ref['password'], http.client.CREATED)

    def get_request_token(self, password, expected_status):
        auth_data = self.build_authentication_request(user_id=self.user_ref['id'], password=password)
        r = self.v3_create_token(auth_data, expected_status=expected_status)
        return r.headers.get('X-Subject-Token')

    def change_password(self, expected_status, **kwargs):
        """Return a test response for a change password request."""
        return self.post('/users/%s/password' % self.user_ref['id'], body={'user': kwargs}, token=self.token, expected_status=expected_status)