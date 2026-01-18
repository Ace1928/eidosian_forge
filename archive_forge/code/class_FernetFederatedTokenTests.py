import copy
import os
import random
import re
import subprocess
from testtools import matchers
from unittest import mock
import uuid
import fixtures
import flask
import http.client
from lxml import etree
from oslo_serialization import jsonutils
from oslo_utils import importutils
import saml2
from saml2 import saml
from saml2 import sigver
import urllib
from keystone.api._shared import authentication
from keystone.api import auth as auth_api
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import render_token
import keystone.conf
from keystone import exception
from keystone.federation import idp as keystone_idp
from keystone.models import token_model
from keystone import notifications
from keystone.tests import unit
from keystone.tests.unit import core
from keystone.tests.unit import federation_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
class FernetFederatedTokenTests(test_v3.RestfulTestCase, FederatedSetupMixin):
    AUTH_METHOD = 'token'

    def load_fixtures(self, fixtures):
        super(FernetFederatedTokenTests, self).load_fixtures(fixtures)
        self.load_federation_sample_data()

    def config_overrides(self):
        super(FernetFederatedTokenTests, self).config_overrides()
        self.config_fixture.config(group='token', provider='fernet')
        self.useFixture(ksfixtures.KeyRepository(self.config_fixture, 'fernet_tokens', CONF.fernet_tokens.max_active_keys))

    def auth_plugin_config_override(self):
        methods = ['saml2', 'token', 'password']
        super(FernetFederatedTokenTests, self).auth_plugin_config_override(methods)

    def test_federated_unscoped_token(self):
        resp = self._issue_unscoped_token()
        self.assertValidMappedUser(render_token.render_token_response_from_model(resp)['token'])

    def test_federated_unscoped_token_with_multiple_groups(self):
        assertion = 'ANOTHER_CUSTOMER_ASSERTION'
        resp = self._issue_unscoped_token(assertion=assertion)
        self.assertValidMappedUser(render_token.render_token_response_from_model(resp)['token'])

    def test_validate_federated_unscoped_token(self):
        resp = self._issue_unscoped_token()
        unscoped_token = resp.id
        self.get('/auth/tokens/', headers={'X-Subject-Token': unscoped_token})

    def test_fernet_full_workflow(self):
        """Test 'standard' workflow for granting Fernet access tokens.

        * Issue unscoped token
        * List available projects based on groups
        * Scope token to one of available projects

        """
        resp = self._issue_unscoped_token()
        self.assertValidMappedUser(render_token.render_token_response_from_model(resp)['token'])
        unscoped_token = resp.id
        resp = self.get('/auth/projects', token=unscoped_token)
        projects = resp.result['projects']
        random_project = random.randint(0, len(projects) - 1)
        project = projects[random_project]
        v3_scope_request = self._scope_request(unscoped_token, 'project', project['id'])
        resp = self.v3_create_token(v3_scope_request)
        token_resp = resp.result['token']
        self._check_project_scoped_token_attributes(token_resp, project['id'])