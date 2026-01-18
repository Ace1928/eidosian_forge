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
class FederatedUserTests(test_v3.RestfulTestCase, FederatedSetupMixin):
    """Test for federated users.

    Tests new shadow users functionality

    """

    def auth_plugin_config_override(self):
        methods = ['saml2', 'token']
        super(FederatedUserTests, self).auth_plugin_config_override(methods)

    def load_fixtures(self, fixtures):
        super(FederatedUserTests, self).load_fixtures(fixtures)
        self.load_federation_sample_data()

    def test_user_id_persistense(self):
        """Ensure user_id is persistend for multiple federated authn calls."""
        r = self._issue_unscoped_token()
        user_id = r.user_id
        self.assertNotEmpty(PROVIDERS.identity_api.get_user(user_id))
        r = self._issue_unscoped_token()
        user_id2 = r.user_id
        self.assertNotEmpty(PROVIDERS.identity_api.get_user(user_id2))
        self.assertEqual(user_id, user_id2)

    def test_user_role_assignment(self):
        project_ref = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
        role_ref = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
        user_id, unscoped_token = self._authenticate_via_saml()
        v3_scope_request = self._scope_request(unscoped_token, 'project', project_ref['id'])
        r = self.v3_create_token(v3_scope_request, expected_status=http.client.UNAUTHORIZED)
        PROVIDERS.assignment_api.add_role_to_user_and_project(user_id, project_ref['id'], role_ref['id'])
        r = self.v3_create_token(v3_scope_request, expected_status=http.client.CREATED)
        scoped_token = r.headers['X-Subject-Token']
        path = '/projects/%(project_id)s' % {'project_id': project_ref['id']}
        r = self.v3_request(path=path, method='GET', expected_status=http.client.OK, token=scoped_token)
        self.assertValidProjectResponse(r, project_ref)
        project_ref2 = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        PROVIDERS.resource_api.create_project(project_ref2['id'], project_ref2)
        path = '/projects/%(project_id)s' % {'project_id': project_ref2['id']}
        r = self.v3_request(path=path, method='GET', expected_status=http.client.FORBIDDEN, token=scoped_token)

    def test_domain_scoped_user_role_assignment(self):
        domain_ref = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain_ref['id'], domain_ref)
        role_ref = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
        user_id, unscoped_token = self._authenticate_via_saml()
        v3_scope_request = self._scope_request(unscoped_token, 'domain', domain_ref['id'])
        r = self.v3_create_token(v3_scope_request, expected_status=http.client.UNAUTHORIZED)
        PROVIDERS.assignment_api.create_grant(role_ref['id'], user_id=user_id, domain_id=domain_ref['id'])
        r = self.v3_create_token(v3_scope_request, expected_status=http.client.CREATED)
        self.assertIsNotNone(r.headers.get('X-Subject-Token'))
        token_resp = r.result['token']
        self.assertIn('domain', token_resp)

    def test_auth_projects_matches_federation_projects(self):
        project_ref = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
        role_ref = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
        user_id, unscoped_token = self._authenticate_via_saml()
        PROVIDERS.assignment_api.add_role_to_user_and_project(user_id, project_ref['id'], role_ref['id'])
        r = self.get('/auth/projects', token=unscoped_token)
        auth_projects = r.result['projects']
        r = self.get('/OS-FEDERATION/projects', token=unscoped_token)
        fed_projects = r.result['projects']
        self.assertCountEqual(auth_projects, fed_projects)

    def test_auth_projects_matches_federation_projects_with_group_assign(self):
        domain_id = CONF.identity.default_domain_id
        project_ref = unit.new_project_ref(domain_id=domain_id)
        PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
        role_ref = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
        group_ref = unit.new_group_ref(domain_id=domain_id)
        group_ref = PROVIDERS.identity_api.create_group(group_ref)
        user_id, unscoped_token = self._authenticate_via_saml()
        PROVIDERS.assignment_api.create_grant(role_ref['id'], group_id=group_ref['id'], project_id=project_ref['id'], domain_id=domain_id)
        PROVIDERS.identity_api.add_user_to_group(user_id=user_id, group_id=group_ref['id'])
        r = self.get('/auth/projects', token=unscoped_token)
        auth_projects = r.result['projects']
        r = self.get('/OS-FEDERATION/projects', token=unscoped_token)
        fed_projects = r.result['projects']
        self.assertCountEqual(auth_projects, fed_projects)

    def test_auth_domains_matches_federation_domains(self):
        domain_ref = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain_ref['id'], domain_ref)
        role_ref = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
        user_id, unscoped_token = self._authenticate_via_saml()
        PROVIDERS.assignment_api.create_grant(role_ref['id'], user_id=user_id, domain_id=domain_ref['id'])
        r = self.get('/auth/domains', token=unscoped_token)
        auth_domains = r.result['domains']
        r = self.get('/OS-FEDERATION/domains', token=unscoped_token)
        fed_domains = r.result['domains']
        self.assertCountEqual(auth_domains, fed_domains)

    def test_auth_domains_matches_federation_domains_with_group_assign(self):
        domain_ref = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain_ref['id'], domain_ref)
        role_ref = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
        group_ref = unit.new_group_ref(domain_id=domain_ref['id'])
        group_ref = PROVIDERS.identity_api.create_group(group_ref)
        user_id, unscoped_token = self._authenticate_via_saml()
        PROVIDERS.assignment_api.create_grant(role_ref['id'], group_id=group_ref['id'], domain_id=domain_ref['id'])
        PROVIDERS.identity_api.add_user_to_group(user_id=user_id, group_id=group_ref['id'])
        r = self.get('/auth/domains', token=unscoped_token)
        auth_domains = r.result['domains']
        r = self.get('/OS-FEDERATION/domains', token=unscoped_token)
        fed_domains = r.result['domains']
        self.assertCountEqual(auth_domains, fed_domains)

    def test_list_head_domains_for_user_duplicates(self):
        role_ref = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
        user_id, unscoped_token = self._authenticate_via_saml()
        r = self.get('/OS-FEDERATION/domains', token=unscoped_token)
        group_domains = r.result['domains']
        domain_from_group = group_domains[0]
        self.head('/OS-FEDERATION/domains', token=unscoped_token, expected_status=http.client.OK)
        PROVIDERS.assignment_api.create_grant(role_ref['id'], user_id=user_id, domain_id=domain_from_group['id'])
        r = self.get('/OS-FEDERATION/domains', token=unscoped_token)
        user_domains = r.result['domains']
        user_domain_ids = []
        for domain in user_domains:
            self.assertNotIn(domain['id'], user_domain_ids)
            user_domain_ids.append(domain['id'])
        r = self.get('/auth/domains', token=unscoped_token)
        user_domains = r.result['domains']
        user_domain_ids = []
        for domain in user_domains:
            self.assertNotIn(domain['id'], user_domain_ids)
            user_domain_ids.append(domain['id'])

    def test_list_head_projects_for_user_duplicates(self):
        role_ref = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
        user_id, unscoped_token = self._authenticate_via_saml()
        r = self.get('/OS-FEDERATION/projects', token=unscoped_token)
        group_projects = r.result['projects']
        project_from_group = group_projects[0]
        self.head('/OS-FEDERATION/projects', token=unscoped_token, expected_status=http.client.OK)
        PROVIDERS.assignment_api.add_role_to_user_and_project(user_id, project_from_group['id'], role_ref['id'])
        r = self.get('/OS-FEDERATION/projects', token=unscoped_token)
        user_projects = r.result['projects']
        user_project_ids = []
        for project in user_projects:
            self.assertNotIn(project['id'], user_project_ids)
            user_project_ids.append(project['id'])
        r = self.get('/auth/projects', token=unscoped_token)
        user_projects = r.result['projects']
        user_project_ids = []
        for project in user_projects:
            self.assertNotIn(project['id'], user_project_ids)
            user_project_ids.append(project['id'])

    def test_delete_protocol_after_federated_authentication(self):
        protocol = self.proto_ref(mapping_id=self.mapping['id'])
        PROVIDERS.federation_api.create_protocol(self.IDP, protocol['id'], protocol)
        r = self._issue_unscoped_token()
        user_id = r.user_id
        self.assertNotEmpty(PROVIDERS.identity_api.get_user(user_id))
        PROVIDERS.federation_api.delete_protocol(self.IDP, protocol['id'])

    def _authenticate_via_saml(self):
        r = self._issue_unscoped_token()
        unscoped_token = r.id
        token_resp = render_token.render_token_response_from_model(r)['token']
        self.assertValidMappedUser(token_resp)
        return (r.user_id, unscoped_token)