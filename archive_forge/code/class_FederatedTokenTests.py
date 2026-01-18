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
class FederatedTokenTests(test_v3.RestfulTestCase, FederatedSetupMixin):

    def auth_plugin_config_override(self):
        methods = ['saml2', 'token']
        super(FederatedTokenTests, self).auth_plugin_config_override(methods)

    def setUp(self):
        super(FederatedTokenTests, self).setUp()
        self._notifications = []

        def fake_saml_notify(action, user_id, group_ids, identity_provider, protocol, token_id, outcome):
            note = {'action': action, 'user_id': user_id, 'identity_provider': identity_provider, 'protocol': protocol, 'send_notification_called': True}
            self._notifications.append(note)
        self.useFixture(fixtures.MockPatchObject(notifications, 'send_saml_audit_notification', fake_saml_notify))

    def _assert_last_notify(self, action, identity_provider, protocol, user_id=None):
        self.assertTrue(self._notifications)
        note = self._notifications[-1]
        if user_id:
            self.assertEqual(note['user_id'], user_id)
        self.assertEqual(note['action'], action)
        self.assertEqual(note['identity_provider'], identity_provider)
        self.assertEqual(note['protocol'], protocol)
        self.assertTrue(note['send_notification_called'])

    def load_fixtures(self, fixtures):
        super(FederatedTokenTests, self).load_fixtures(fixtures)
        self.load_federation_sample_data()

    def test_issue_unscoped_token_notify(self):
        self._issue_unscoped_token()
        self._assert_last_notify(self.ACTION, self.IDP, self.PROTOCOL)

    def test_issue_unscoped_token(self):
        r = self._issue_unscoped_token()
        token_resp = render_token.render_token_response_from_model(r)['token']
        self.assertValidMappedUser(token_resp)

    def test_default_domain_scoped_token(self):
        self.config_fixture.config(group='token', cache_on_issue=False)
        token = self._issue_unscoped_token()
        PROVIDERS.assignment_api.create_grant(self.role_admin['id'], user_id=token.user_id, domain_id=CONF.identity.default_domain_id)
        auth_request = {'auth': {'identity': {'methods': ['token'], 'token': {'id': token.id}}, 'scope': {'domain': {'id': CONF.identity.default_domain_id}}}}
        r = self.v3_create_token(auth_request)
        domain_scoped_token_id = r.headers.get('X-Subject-Token')
        headers = {'X-Subject-Token': domain_scoped_token_id}
        self.get('/auth/tokens', token=domain_scoped_token_id, headers=headers)

    def test_issue_the_same_unscoped_token_with_user_deleted(self):
        r = self._issue_unscoped_token()
        token = render_token.render_token_response_from_model(r)['token']
        user1 = token['user']
        user_id1 = user1.pop('id')
        PROVIDERS.identity_api.delete_user(user_id1)
        r = self._issue_unscoped_token()
        token = render_token.render_token_response_from_model(r)['token']
        user2 = token['user']
        user_id2 = user2.pop('id')
        self.assertIsNot(user_id2, user_id1)
        self.assertEqual(user1, user2)

    def test_issue_unscoped_token_disabled_idp(self):
        """Check if authentication works with disabled identity providers.

        Test plan:
        1) Disable default IdP
        2) Try issuing unscoped token for that IdP
        3) Expect server to forbid authentication

        """
        enabled_false = {'enabled': False}
        PROVIDERS.federation_api.update_idp(self.IDP, enabled_false)
        self.assertRaises(exception.Forbidden, self._issue_unscoped_token)

    def test_issue_unscoped_token_group_names_in_mapping(self):
        r = self._issue_unscoped_token(assertion='ANOTHER_CUSTOMER_ASSERTION')
        ref_groups = set([self.group_customers['id'], self.group_admins['id']])
        token_groups = r.federated_groups
        token_groups = set([group['id'] for group in token_groups])
        self.assertEqual(ref_groups, token_groups)

    def test_issue_unscoped_tokens_nonexisting_group(self):
        self._issue_unscoped_token(assertion='ANOTHER_TESTER_ASSERTION')

    def test_issue_unscoped_token_with_remote_no_attribute(self):
        self._issue_unscoped_token(idp=self.IDP_WITH_REMOTE, environment={self.REMOTE_ID_ATTR: self.REMOTE_IDS[0]})

    def test_issue_unscoped_token_with_remote(self):
        self.config_fixture.config(group='federation', remote_id_attribute=self.REMOTE_ID_ATTR)
        self._issue_unscoped_token(idp=self.IDP_WITH_REMOTE, environment={self.REMOTE_ID_ATTR: self.REMOTE_IDS[0]})

    def test_issue_unscoped_token_with_saml2_remote(self):
        self.config_fixture.config(group='saml2', remote_id_attribute=self.REMOTE_ID_ATTR)
        self._issue_unscoped_token(idp=self.IDP_WITH_REMOTE, environment={self.REMOTE_ID_ATTR: self.REMOTE_IDS[0]})

    def test_issue_unscoped_token_with_remote_different(self):
        self.config_fixture.config(group='federation', remote_id_attribute=self.REMOTE_ID_ATTR)
        self.assertRaises(exception.Forbidden, self._issue_unscoped_token, idp=self.IDP_WITH_REMOTE, environment={self.REMOTE_ID_ATTR: uuid.uuid4().hex})

    def test_issue_unscoped_token_with_remote_default_overwritten(self):
        """Test that protocol remote_id_attribute has higher priority.

        Make sure the parameter stored under ``protocol`` section has higher
        priority over parameter from default ``federation`` configuration
        section.

        """
        self.config_fixture.config(group='saml2', remote_id_attribute=self.REMOTE_ID_ATTR)
        self.config_fixture.config(group='federation', remote_id_attribute=uuid.uuid4().hex)
        self._issue_unscoped_token(idp=self.IDP_WITH_REMOTE, environment={self.REMOTE_ID_ATTR: self.REMOTE_IDS[0]})

    def test_issue_unscoped_token_with_remote_unavailable(self):
        self.config_fixture.config(group='federation', remote_id_attribute=self.REMOTE_ID_ATTR)
        self.assertRaises(exception.Unauthorized, self._issue_unscoped_token, idp=self.IDP_WITH_REMOTE, environment={uuid.uuid4().hex: uuid.uuid4().hex})

    def test_issue_unscoped_token_with_remote_user_as_empty_string(self):
        self._issue_unscoped_token(environment={'REMOTE_USER': ''})

    def test_issue_unscoped_token_no_groups(self):
        r = self._issue_unscoped_token(assertion='USER_NO_GROUPS_ASSERTION')
        token_groups = r.federated_groups
        self.assertEqual(0, len(token_groups))

    def test_issue_scoped_token_no_groups(self):
        """Verify that token without groups cannot get scoped to project.

        This test is required because of bug 1677723.
        """
        r = self._issue_unscoped_token(assertion='USER_NO_GROUPS_ASSERTION')
        token_groups = r.federated_groups
        self.assertEqual(0, len(token_groups))
        unscoped_token = r.id
        self.proj_employees
        admin = unit.new_user_ref(CONF.identity.default_domain_id)
        PROVIDERS.identity_api.create_user(admin)
        PROVIDERS.assignment_api.create_grant(self.role_admin['id'], user_id=admin['id'], project_id=self.proj_employees['id'])
        scope = self._scope_request(unscoped_token, 'project', self.proj_employees['id'])
        self.v3_create_token(scope, expected_status=http.client.UNAUTHORIZED)

    def test_issue_unscoped_token_malformed_environment(self):
        """Test whether non string objects are filtered out.

        Put non string objects into the environment, inject
        correct assertion and try to get an unscoped token.
        Expect server not to fail on using split() method on
        non string objects and return token id in the HTTP header.

        """
        environ = {'malformed_object': object(), 'another_bad_idea': tuple(range(10)), 'yet_another_bad_param': dict(zip(uuid.uuid4().hex, range(32)))}
        environ.update(mapping_fixtures.EMPLOYEE_ASSERTION)
        with self.make_request(environ=environ):
            authentication.authenticate_for_token(self.UNSCOPED_V3_SAML2_REQ)

    def test_scope_to_project_once_notify(self):
        r = self.v3_create_token(self.TOKEN_SCOPE_PROJECT_EMPLOYEE_FROM_EMPLOYEE)
        user_id = r.json['token']['user']['id']
        self._assert_last_notify(self.ACTION, self.IDP, self.PROTOCOL, user_id)

    def test_scope_to_project_once(self):
        r = self.v3_create_token(self.TOKEN_SCOPE_PROJECT_EMPLOYEE_FROM_EMPLOYEE)
        token_resp = r.result['token']
        project_id = token_resp['project']['id']
        self._check_project_scoped_token_attributes(token_resp, project_id)
        roles_ref = [self.role_employee]
        projects_ref = self.proj_employees
        self._check_projects_and_roles(token_resp, roles_ref, projects_ref)
        self.assertValidMappedUser(token_resp)

    def test_scope_token_with_idp_disabled(self):
        """Scope token issued by disabled IdP.

        Try scoping the token issued by an IdP which is disabled now. Expect
        server to refuse scoping operation.

        This test confirms correct behaviour when IdP was enabled and unscoped
        token was issued, but disabled before user tries to scope the token.
        Here we assume the unscoped token was already issued and start from
        the moment where IdP is being disabled and unscoped token is being
        used.

        Test plan:
        1) Disable IdP
        2) Try scoping unscoped token

        """
        enabled_false = {'enabled': False}
        PROVIDERS.federation_api.update_idp(self.IDP, enabled_false)
        self.v3_create_token(self.TOKEN_SCOPE_PROJECT_EMPLOYEE_FROM_CUSTOMER, expected_status=http.client.FORBIDDEN)

    def test_validate_token_after_deleting_idp_raises_not_found(self):
        token = self.v3_create_token(self.TOKEN_SCOPE_PROJECT_EMPLOYEE_FROM_ADMIN)
        token_id = token.headers.get('X-Subject-Token')
        federated_info = token.json_body['token']['user']['OS-FEDERATION']
        idp_id = federated_info['identity_provider']['id']
        PROVIDERS.federation_api.delete_idp(idp_id)
        headers = {'X-Subject-Token': token_id}
        self.get('/auth/tokens/', token=token_id, headers=headers, expected_status=http.client.NOT_FOUND)

    def test_deleting_idp_cascade_deleting_fed_user(self):
        token = self.v3_create_token(self.TOKEN_SCOPE_PROJECT_EMPLOYEE_FROM_ADMIN)
        federated_info = token.json_body['token']['user']['OS-FEDERATION']
        idp_id = federated_info['identity_provider']['id']
        hints = driver_hints.Hints()
        hints.add_filter('idp_id', idp_id)
        fed_users = PROVIDERS.shadow_users_api.get_federated_users(hints)
        self.assertEqual(3, len(fed_users))
        idp_domain_id = PROVIDERS.federation_api.get_idp(idp_id)['domain_id']
        for fed_user in fed_users:
            self.assertEqual(idp_domain_id, fed_user['domain_id'])
        PROVIDERS.federation_api.delete_idp(idp_id)
        hints = driver_hints.Hints()
        hints.add_filter('idp_id', idp_id)
        fed_users = PROVIDERS.shadow_users_api.get_federated_users(hints)
        self.assertEqual([], fed_users)

    def test_scope_to_bad_project(self):
        """Scope unscoped token with a project we don't have access to."""
        self.v3_create_token(self.TOKEN_SCOPE_PROJECT_EMPLOYEE_FROM_CUSTOMER, expected_status=http.client.UNAUTHORIZED)

    def test_scope_to_project_multiple_times(self):
        """Try to scope the unscoped token multiple times.

        The new tokens should be scoped to:

        * Customers' project
        * Employees' project

        """
        bodies = (self.TOKEN_SCOPE_PROJECT_EMPLOYEE_FROM_ADMIN, self.TOKEN_SCOPE_PROJECT_CUSTOMER_FROM_ADMIN)
        project_ids = (self.proj_employees['id'], self.proj_customers['id'])
        for body, project_id_ref in zip(bodies, project_ids):
            r = self.v3_create_token(body)
            token_resp = r.result['token']
            self._check_project_scoped_token_attributes(token_resp, project_id_ref)

    def test_scope_to_project_with_duplicate_roles_returns_single_role(self):
        r = self.v3_create_token(self.TOKEN_SCOPE_PROJECT_EMPLOYEE_FROM_ADMIN)
        user_id = r.json_body['token']['user']['id']
        project_id = r.json_body['token']['project']['id']
        for role in r.json_body['token']['roles']:
            PROVIDERS.assignment_api.create_grant(role_id=role['id'], user_id=user_id, project_id=project_id)
        r = self.v3_create_token(self.TOKEN_SCOPE_PROJECT_EMPLOYEE_FROM_ADMIN)
        known_role_ids = []
        for role in r.json_body['token']['roles']:
            self.assertNotIn(role['id'], known_role_ids)
            known_role_ids.append(role['id'])

    def test_scope_to_project_with_only_inherited_roles(self):
        """Try to scope token whose only roles are inherited."""
        r = self.v3_create_token(self.TOKEN_SCOPE_PROJECT_INHERITED_FROM_CUSTOMER)
        token_resp = r.result['token']
        self._check_project_scoped_token_attributes(token_resp, self.project_inherited['id'])
        roles_ref = [self.role_customer]
        projects_ref = self.project_inherited
        self._check_projects_and_roles(token_resp, roles_ref, projects_ref)
        self.assertValidMappedUser(token_resp)

    def test_scope_token_from_nonexistent_unscoped_token(self):
        """Try to scope token from non-existent unscoped token."""
        self.v3_create_token(self.TOKEN_SCOPE_PROJECT_FROM_NONEXISTENT_TOKEN, expected_status=http.client.NOT_FOUND)

    def test_issue_token_from_rules_without_user(self):
        environ = copy.deepcopy(mapping_fixtures.BAD_TESTER_ASSERTION)
        with self.make_request(environ=environ):
            self.assertRaises(exception.Unauthorized, authentication.authenticate_for_token, self.UNSCOPED_V3_SAML2_REQ)

    def test_issue_token_with_nonexistent_group(self):
        """Inject assertion that matches rule issuing bad group id.

        Expect server to find out that some groups are missing in the
        backend and raise exception.MappedGroupNotFound exception.

        """
        self.assertRaises(exception.MappedGroupNotFound, self._issue_unscoped_token, assertion='CONTRACTOR_ASSERTION')

    def test_scope_to_domain_once(self):
        r = self.v3_create_token(self.TOKEN_SCOPE_DOMAIN_A_FROM_CUSTOMER)
        token_resp = r.result['token']
        self._check_domain_scoped_token_attributes(token_resp, self.domainA['id'])

    def test_scope_to_domain_multiple_tokens(self):
        """Issue multiple tokens scoping to different domains.

        The new tokens should be scoped to:

        * domainA
        * domainB
        * domainC

        """
        bodies = (self.TOKEN_SCOPE_DOMAIN_A_FROM_ADMIN, self.TOKEN_SCOPE_DOMAIN_B_FROM_ADMIN, self.TOKEN_SCOPE_DOMAIN_C_FROM_ADMIN)
        domain_ids = (self.domainA['id'], self.domainB['id'], self.domainC['id'])
        for body, domain_id_ref in zip(bodies, domain_ids):
            r = self.v3_create_token(body)
            token_resp = r.result['token']
            self._check_domain_scoped_token_attributes(token_resp, domain_id_ref)

    def test_scope_to_domain_with_only_inherited_roles_fails(self):
        """Try to scope to a domain that has no direct roles."""
        self.v3_create_token(self.TOKEN_SCOPE_DOMAIN_D_FROM_CUSTOMER, expected_status=http.client.UNAUTHORIZED)

    def test_list_projects(self):
        urls = ('/OS-FEDERATION/projects', '/auth/projects')
        token = (self.tokens['CUSTOMER_ASSERTION'], self.tokens['EMPLOYEE_ASSERTION'], self.tokens['ADMIN_ASSERTION'])
        projects_refs = (set([self.proj_customers['id'], self.project_inherited['id']]), set([self.proj_employees['id'], self.project_all['id']]), set([self.proj_employees['id'], self.project_all['id'], self.proj_customers['id'], self.project_inherited['id']]))
        for token, projects_ref in zip(token, projects_refs):
            for url in urls:
                r = self.get(url, token=token)
                projects_resp = r.result['projects']
                projects = set((p['id'] for p in projects_resp))
                self.assertEqual(projects_ref, projects, 'match failed for url %s' % url)

    def test_list_projects_for_inherited_project_assignment(self):
        subproject_inherited = unit.new_project_ref(domain_id=self.domainD['id'], parent_id=self.project_inherited['id'])
        PROVIDERS.resource_api.create_project(subproject_inherited['id'], subproject_inherited)
        PROVIDERS.assignment_api.create_grant(role_id=self.role_employee['id'], group_id=self.group_employees['id'], project_id=self.project_inherited['id'], inherited_to_projects=True)
        expected_project_ids = [self.project_all['id'], self.proj_employees['id'], subproject_inherited['id']]
        for url in ('/OS-FEDERATION/projects', '/auth/projects'):
            r = self.get(url, token=self.tokens['EMPLOYEE_ASSERTION'])
            project_ids = [project['id'] for project in r.result['projects']]
            self.assertEqual(len(expected_project_ids), len(project_ids))
            for expected_project_id in expected_project_ids:
                self.assertIn(expected_project_id, project_ids, 'Projects match failed for url %s' % url)

    def test_list_domains(self):
        urls = ('/OS-FEDERATION/domains', '/auth/domains')
        tokens = (self.tokens['CUSTOMER_ASSERTION'], self.tokens['EMPLOYEE_ASSERTION'], self.tokens['ADMIN_ASSERTION'])
        domain_refs = (set([self.domainA['id']]), set([self.domainA['id'], self.domainB['id']]), set([self.domainA['id'], self.domainB['id'], self.domainC['id']]))
        for token, domains_ref in zip(tokens, domain_refs):
            for url in urls:
                r = self.get(url, token=token)
                domains_resp = r.result['domains']
                domains = set((p['id'] for p in domains_resp))
                self.assertEqual(domains_ref, domains, 'match failed for url %s' % url)

    def test_full_workflow(self):
        """Test 'standard' workflow for granting access tokens.

        * Issue unscoped token
        * List available projects based on groups
        * Scope token to one of available projects

        """
        r = self._issue_unscoped_token()
        token_resp = render_token.render_token_response_from_model(r)['token']
        self.assertListEqual(['saml2'], r.methods)
        self.assertValidMappedUser(token_resp)
        employee_unscoped_token_id = r.id
        r = self.get('/auth/projects', token=employee_unscoped_token_id)
        projects = r.result['projects']
        random_project = random.randint(0, len(projects) - 1)
        project = projects[random_project]
        v3_scope_request = self._scope_request(employee_unscoped_token_id, 'project', project['id'])
        r = self.v3_create_token(v3_scope_request)
        token_resp = r.result['token']
        self.assertIn('token', token_resp['methods'])
        self.assertIn('saml2', token_resp['methods'])
        self._check_project_scoped_token_attributes(token_resp, project['id'])

    def test_workflow_with_groups_deletion(self):
        """Test full workflow with groups deletion before token scoping.

        The test scenario is as follows:
         - Create group ``group``
         - Create and assign roles to ``group`` and ``project_all``
         - Patch mapping rules for existing IdP so it issues group id
         - Issue unscoped token with ``group``'s id
         - Delete group ``group``
         - Scope token to ``project_all``
         - Expect HTTP 500 response

        """
        group = unit.new_group_ref(domain_id=self.domainA['id'])
        group = PROVIDERS.identity_api.create_group(group)
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        PROVIDERS.assignment_api.create_grant(role['id'], group_id=group['id'], project_id=self.project_all['id'])
        rules = {'rules': [{'local': [{'group': {'id': group['id']}}, {'user': {'name': '{0}'}}], 'remote': [{'type': 'UserName'}, {'type': 'LastName', 'any_one_of': ['Account']}]}]}
        PROVIDERS.federation_api.update_mapping(self.mapping['id'], rules)
        r = self._issue_unscoped_token(assertion='TESTER_ASSERTION')
        PROVIDERS.identity_api.delete_group(group['id'])
        scoped_token = self._scope_request(r.id, 'project', self.project_all['id'])
        self.v3_create_token(scoped_token, expected_status=http.client.INTERNAL_SERVER_ERROR)

    def test_lists_with_missing_group_in_backend(self):
        """Test a mapping that points to a group that does not exist.

        For explicit mappings, we expect the group to exist in the backend,
        but for lists, specifically blacklists, a missing group is expected
        as many groups will be specified by the IdP that are not Keystone
        groups.

        The test scenario is as follows:
         - Create group ``EXISTS``
         - Set mapping rules for existing IdP with a blacklist
           that passes through as REMOTE_USER_GROUPS
         - Issue unscoped token with on group  ``EXISTS`` id in it

        """
        domain_id = self.domainA['id']
        domain_name = self.domainA['name']
        group = unit.new_group_ref(domain_id=domain_id, name='EXISTS')
        group = PROVIDERS.identity_api.create_group(group)
        rules = {'rules': [{'local': [{'user': {'name': '{0}', 'id': '{0}'}}], 'remote': [{'type': 'REMOTE_USER'}]}, {'local': [{'groups': '{0}', 'domain': {'name': domain_name}}], 'remote': [{'type': 'REMOTE_USER_GROUPS'}]}]}
        PROVIDERS.federation_api.update_mapping(self.mapping['id'], rules)
        r = self._issue_unscoped_token(assertion='UNMATCHED_GROUP_ASSERTION')
        assigned_group_ids = r.federated_groups
        self.assertEqual(1, len(assigned_group_ids))
        self.assertEqual(group['id'], assigned_group_ids[0]['id'])

    def test_empty_blacklist_passess_all_values(self):
        """Test a mapping with empty blacklist specified.

        Not adding a ``blacklist`` keyword to the mapping rules has the same
        effect as adding an empty ``blacklist``.
        In both cases, the mapping engine will not discard any groups that are
        associated with apache environment variables.

        This test checks scenario where an empty blacklist was specified.
        Expected result is to allow any value.

        The test scenario is as follows:
         - Create group ``EXISTS``
         - Create group ``NO_EXISTS``
         - Set mapping rules for existing IdP with a blacklist
           that passes through as REMOTE_USER_GROUPS
         - Issue unscoped token with groups  ``EXISTS`` and ``NO_EXISTS``
           assigned

        """
        domain_id = self.domainA['id']
        domain_name = self.domainA['name']
        group_exists = unit.new_group_ref(domain_id=domain_id, name='EXISTS')
        group_exists = PROVIDERS.identity_api.create_group(group_exists)
        group_no_exists = unit.new_group_ref(domain_id=domain_id, name='NO_EXISTS')
        group_no_exists = PROVIDERS.identity_api.create_group(group_no_exists)
        group_ids = set([group_exists['id'], group_no_exists['id']])
        rules = {'rules': [{'local': [{'user': {'name': '{0}', 'id': '{0}'}}], 'remote': [{'type': 'REMOTE_USER'}]}, {'local': [{'groups': '{0}', 'domain': {'name': domain_name}}], 'remote': [{'type': 'REMOTE_USER_GROUPS', 'blacklist': []}]}]}
        PROVIDERS.federation_api.update_mapping(self.mapping['id'], rules)
        r = self._issue_unscoped_token(assertion='UNMATCHED_GROUP_ASSERTION')
        assigned_group_ids = r.federated_groups
        self.assertEqual(len(group_ids), len(assigned_group_ids))
        for group in assigned_group_ids:
            self.assertIn(group['id'], group_ids)

    def test_not_adding_blacklist_passess_all_values(self):
        """Test a mapping without blacklist specified.

        Not adding a ``blacklist`` keyword to the mapping rules has the same
        effect as adding an empty ``blacklist``. In both cases all values will
        be accepted and passed.

        This test checks scenario where an blacklist was not specified.
        Expected result is to allow any value.

        The test scenario is as follows:
         - Create group ``EXISTS``
         - Create group ``NO_EXISTS``
         - Set mapping rules for existing IdP with a blacklist
           that passes through as REMOTE_USER_GROUPS
         - Issue unscoped token with on groups ``EXISTS`` and ``NO_EXISTS``
           assigned

        """
        domain_id = self.domainA['id']
        domain_name = self.domainA['name']
        group_exists = unit.new_group_ref(domain_id=domain_id, name='EXISTS')
        group_exists = PROVIDERS.identity_api.create_group(group_exists)
        group_no_exists = unit.new_group_ref(domain_id=domain_id, name='NO_EXISTS')
        group_no_exists = PROVIDERS.identity_api.create_group(group_no_exists)
        group_ids = set([group_exists['id'], group_no_exists['id']])
        rules = {'rules': [{'local': [{'user': {'name': '{0}', 'id': '{0}'}}], 'remote': [{'type': 'REMOTE_USER'}]}, {'local': [{'groups': '{0}', 'domain': {'name': domain_name}}], 'remote': [{'type': 'REMOTE_USER_GROUPS'}]}]}
        PROVIDERS.federation_api.update_mapping(self.mapping['id'], rules)
        r = self._issue_unscoped_token(assertion='UNMATCHED_GROUP_ASSERTION')
        assigned_group_ids = r.federated_groups
        self.assertEqual(len(group_ids), len(assigned_group_ids))
        for group in assigned_group_ids:
            self.assertIn(group['id'], group_ids)

    def test_empty_whitelist_discards_all_values(self):
        """Test that empty whitelist blocks all the values.

        Not adding a ``whitelist`` keyword to the mapping value is different
        than adding empty whitelist.  The former case will simply pass all the
        values, whereas the latter would discard all the values.

        This test checks scenario where an empty whitelist was specified.
        The expected result is that no groups are matched.

        The test scenario is as follows:
         - Create group ``EXISTS``
         - Set mapping rules for existing IdP with an empty whitelist
           that whould discard any values from the assertion
         - Try issuing unscoped token, no groups were matched and that the
           federated user does not have any group assigned.

        """
        domain_id = self.domainA['id']
        domain_name = self.domainA['name']
        group = unit.new_group_ref(domain_id=domain_id, name='EXISTS')
        group = PROVIDERS.identity_api.create_group(group)
        rules = {'rules': [{'local': [{'user': {'name': '{0}', 'id': '{0}'}}], 'remote': [{'type': 'REMOTE_USER'}]}, {'local': [{'groups': '{0}', 'domain': {'name': domain_name}}], 'remote': [{'type': 'REMOTE_USER_GROUPS', 'whitelist': []}]}]}
        PROVIDERS.federation_api.update_mapping(self.mapping['id'], rules)
        r = self._issue_unscoped_token(assertion='UNMATCHED_GROUP_ASSERTION')
        assigned_groups = r.federated_groups
        self.assertEqual(len(assigned_groups), 0)

    def test_not_setting_whitelist_accepts_all_values(self):
        """Test that not setting whitelist passes.

        Not adding a ``whitelist`` keyword to the mapping value is different
        than adding empty whitelist.  The former case will simply pass all the
        values, whereas the latter would discard all the values.

        This test checks a scenario where a ``whitelist`` was not specified.
        Expected result is that no groups are ignored.

        The test scenario is as follows:
         - Create group ``EXISTS``
         - Set mapping rules for existing IdP with an empty whitelist
           that whould discard any values from the assertion
         - Issue an unscoped token and make sure ephemeral user is a member of
           two groups.

        """
        domain_id = self.domainA['id']
        domain_name = self.domainA['name']
        group_exists = unit.new_group_ref(domain_id=domain_id, name='EXISTS')
        group_exists = PROVIDERS.identity_api.create_group(group_exists)
        group_no_exists = unit.new_group_ref(domain_id=domain_id, name='NO_EXISTS')
        group_no_exists = PROVIDERS.identity_api.create_group(group_no_exists)
        group_ids = set([group_exists['id'], group_no_exists['id']])
        rules = {'rules': [{'local': [{'user': {'name': '{0}', 'id': '{0}'}}], 'remote': [{'type': 'REMOTE_USER'}]}, {'local': [{'groups': '{0}', 'domain': {'name': domain_name}}], 'remote': [{'type': 'REMOTE_USER_GROUPS'}]}]}
        PROVIDERS.federation_api.update_mapping(self.mapping['id'], rules)
        r = self._issue_unscoped_token(assertion='UNMATCHED_GROUP_ASSERTION')
        assigned_group_ids = r.federated_groups
        self.assertEqual(len(group_ids), len(assigned_group_ids))
        for group in assigned_group_ids:
            self.assertIn(group['id'], group_ids)

    def test_assertion_prefix_parameter(self):
        """Test parameters filtering based on the prefix.

        With ``assertion_prefix`` set to fixed, non default value,
        issue an unscoped token from assertion EMPLOYEE_ASSERTION_PREFIXED.
        Expect server to return unscoped token.

        """
        self.config_fixture.config(group='federation', assertion_prefix=self.ASSERTION_PREFIX)
        self._issue_unscoped_token(assertion='EMPLOYEE_ASSERTION_PREFIXED')

    def test_assertion_prefix_parameter_expect_fail(self):
        """Test parameters filtering based on the prefix.

        With ``assertion_prefix`` default value set to empty string
        issue an unscoped token from assertion EMPLOYEE_ASSERTION.
        Next, configure ``assertion_prefix`` to value ``UserName``.
        Try issuing unscoped token with EMPLOYEE_ASSERTION.
        Expect server to raise exception.Unathorized exception.

        """
        self._issue_unscoped_token()
        self.config_fixture.config(group='federation', assertion_prefix='UserName')
        self.assertRaises(exception.Unauthorized, self._issue_unscoped_token)

    def test_unscoped_token_has_user_domain(self):
        r = self._issue_unscoped_token()
        self._check_domains_are_valid(render_token.render_token_response_from_model(r)['token'])

    def test_scoped_token_has_user_domain(self):
        r = self.v3_create_token(self.TOKEN_SCOPE_PROJECT_EMPLOYEE_FROM_EMPLOYEE)
        self._check_domains_are_valid(r.json_body['token'])

    def test_issue_unscoped_token_for_local_user(self):
        r = self._issue_unscoped_token(assertion='LOCAL_USER_ASSERTION')
        self.assertListEqual(['saml2'], r.methods)
        self.assertEqual(self.user['id'], r.user_id)
        self.assertEqual(self.user['name'], r.user['name'])
        self.assertEqual(self.domain['id'], r.user_domain['id'])
        self.assertIsNone(r.domain_id)
        self.assertIsNone(r.project_id)
        self.assertTrue(r.unscoped)

    def test_issue_token_for_local_user_user_not_found(self):
        self.assertRaises(exception.Unauthorized, self._issue_unscoped_token, assertion='ANOTHER_LOCAL_USER_ASSERTION')

    def test_user_name_and_id_in_federation_token(self):
        r = self._issue_unscoped_token(assertion='EMPLOYEE_ASSERTION')
        self.assertEqual(mapping_fixtures.EMPLOYEE_ASSERTION['UserName'], r.user['name'])
        self.assertNotEqual(r.user['name'], r.user_id)
        r = self.v3_create_token(self.TOKEN_SCOPE_PROJECT_EMPLOYEE_FROM_EMPLOYEE)
        token = r.json_body['token']
        self.assertEqual(mapping_fixtures.EMPLOYEE_ASSERTION['UserName'], token['user']['name'])
        self.assertNotEqual(token['user']['name'], token['user']['id'])

    def test_issue_unscoped_token_with_remote_different_from_protocol(self):
        protocol = PROVIDERS.federation_api.get_protocol(self.IDP_WITH_REMOTE, self.PROTOCOL)
        protocol['remote_id_attribute'] = uuid.uuid4().hex
        PROVIDERS.federation_api.update_protocol(self.IDP_WITH_REMOTE, protocol['id'], protocol)
        self._issue_unscoped_token(idp=self.IDP_WITH_REMOTE, environment={protocol['remote_id_attribute']: self.REMOTE_IDS[0]})
        self.assertRaises(exception.Unauthorized, self._issue_unscoped_token, idp=self.IDP_WITH_REMOTE, environment={uuid.uuid4().hex: self.REMOTE_IDS[0]})

    def test_issue_token_for_ephemeral_user_with_remote_domain(self):
        """Test ephemeral user is created in the domain set by assertion.

        Shadow user may belong to the domain set by the assertion data.
        To verify that:
         - precreate domain used later in the assertion
         - update mapping to unclude user domain name coming from assertion
         - auth user
         - verify user domain is not the IDP domain

        """
        domain_ref = unit.new_domain_ref(name='user_domain')
        PROVIDERS.resource_api.create_domain(domain_ref['id'], domain_ref)
        PROVIDERS.federation_api.update_mapping(self.mapping['id'], mapping_fixtures.MAPPING_EPHEMERAL_USER_REMOTE_DOMAIN)
        r = self._issue_unscoped_token(assertion='USER_WITH_DOMAIN_ASSERTION')
        self.assertEqual(r.user_domain['id'], domain_ref['id'])
        self.assertNotEqual(r.user_domain['id'], self.idp['domain_id'])