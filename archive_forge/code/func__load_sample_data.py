import copy
import hashlib
from unittest import mock
import uuid
import fixtures
import http.client
import webtest
from keystone.auth import core as auth_core
from keystone.common import authorization
from keystone.common import context as keystone_context
from keystone.common import provider_api
from keystone.common import tokenless_auth
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_backend_sql
def _load_sample_data(self):
    self.protocol_id = 'x509'
    self.domain = unit.new_domain_ref()
    self.domain_id = self.domain['id']
    self.domain_name = self.domain['name']
    PROVIDERS.resource_api.create_domain(self.domain_id, self.domain)
    self.project = unit.new_project_ref(domain_id=self.domain_id)
    self.project_id = self.project['id']
    self.project_name = self.project['name']
    PROVIDERS.resource_api.create_project(self.project_id, self.project)
    self.user = unit.new_user_ref(domain_id=self.domain_id, project_id=self.project_id)
    self.user = PROVIDERS.identity_api.create_user(self.user)
    self.idp = self._idp_ref(id=self.idp_id)
    PROVIDERS.federation_api.create_idp(self.idp['id'], self.idp)
    self.role = unit.new_role_ref()
    self.role_id = self.role['id']
    self.role_name = self.role['name']
    PROVIDERS.role_api.create_role(self.role_id, self.role)
    self.group = unit.new_group_ref(domain_id=self.domain_id)
    self.group = PROVIDERS.identity_api.create_group(self.group)
    PROVIDERS.assignment_api.add_role_to_user_and_project(user_id=self.user['id'], project_id=self.project_id, role_id=self.role_id)
    PROVIDERS.assignment_api.create_grant(role_id=self.role_id, group_id=self.group['id'], project_id=self.project_id)