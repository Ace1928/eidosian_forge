import copy
import datetime
import fixtures
import itertools
import operator
import re
from unittest import mock
from urllib import parse
import uuid
from cryptography.hazmat.primitives.serialization import Encoding
import freezegun
import http.client
from oslo_serialization import jsonutils as json
from oslo_utils import fixture
from oslo_utils import timeutils
from testtools import matchers
from testtools import testcase
from keystone import auth
from keystone.auth.plugins import totp
from keystone.common import authorization
from keystone.common import provider_api
from keystone.common.rbac_enforcer import policy
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import resource_options as ro
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
class TestTokenRevokeById(test_v3.RestfulTestCase):
    """Test token revocation on the v3 Identity API."""

    def config_overrides(self):
        super(TestTokenRevokeById, self).config_overrides()
        self.config_fixture.config(group='token', provider='fernet', revoke_by_id=False)
        self.useFixture(ksfixtures.KeyRepository(self.config_fixture, 'fernet_tokens', CONF.fernet_tokens.max_active_keys))

    def setUp(self):
        """Setup for Token Revoking Test Cases.

        As well as the usual housekeeping, create a set of domains,
        users, groups, roles and projects for the subsequent tests:

        - Two domains: A & B
        - Three users (1, 2 and 3)
        - Three groups (1, 2 and 3)
        - Two roles (1 and 2)
        - DomainA owns user1, domainB owns user2 and user3
        - DomainA owns group1 and group2, domainB owns group3
        - User1 and user2 are members of group1
        - User3 is a member of group2
        - Two projects: A & B, both in domainA
        - Group1 has role1 on Project A and B, meaning that user1 and user2
          will get these roles by virtue of membership
        - User1, 2 and 3 have role1 assigned to projectA
        - Group1 has role1 on Project A and B, meaning that user1 and user2
          will get role1 (duplicated) by virtue of membership
        - User1 has role2 assigned to domainA

        """
        super(TestTokenRevokeById, self).setUp()
        self.domainA = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(self.domainA['id'], self.domainA)
        self.domainB = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(self.domainB['id'], self.domainB)
        self.projectA = unit.new_project_ref(domain_id=self.domainA['id'])
        PROVIDERS.resource_api.create_project(self.projectA['id'], self.projectA)
        self.projectB = unit.new_project_ref(domain_id=self.domainA['id'])
        PROVIDERS.resource_api.create_project(self.projectB['id'], self.projectB)
        self.user1 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domainA['id'])
        self.user2 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domainB['id'])
        self.user3 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domainB['id'])
        self.group1 = unit.new_group_ref(domain_id=self.domainA['id'])
        self.group1 = PROVIDERS.identity_api.create_group(self.group1)
        self.group2 = unit.new_group_ref(domain_id=self.domainA['id'])
        self.group2 = PROVIDERS.identity_api.create_group(self.group2)
        self.group3 = unit.new_group_ref(domain_id=self.domainB['id'])
        self.group3 = PROVIDERS.identity_api.create_group(self.group3)
        PROVIDERS.identity_api.add_user_to_group(self.user1['id'], self.group1['id'])
        PROVIDERS.identity_api.add_user_to_group(self.user2['id'], self.group1['id'])
        PROVIDERS.identity_api.add_user_to_group(self.user3['id'], self.group2['id'])
        self.role1 = unit.new_role_ref()
        PROVIDERS.role_api.create_role(self.role1['id'], self.role1)
        self.role2 = unit.new_role_ref()
        PROVIDERS.role_api.create_role(self.role2['id'], self.role2)
        PROVIDERS.assignment_api.create_grant(self.role2['id'], user_id=self.user1['id'], domain_id=self.domainA['id'])
        PROVIDERS.assignment_api.create_grant(self.role1['id'], user_id=self.user1['id'], project_id=self.projectA['id'])
        PROVIDERS.assignment_api.create_grant(self.role1['id'], user_id=self.user2['id'], project_id=self.projectA['id'])
        PROVIDERS.assignment_api.create_grant(self.role1['id'], user_id=self.user3['id'], project_id=self.projectA['id'])
        PROVIDERS.assignment_api.create_grant(self.role1['id'], group_id=self.group1['id'], project_id=self.projectA['id'])

    def test_unscoped_token_remains_valid_after_role_assignment(self):
        unscoped_token = self.get_requested_token(self.build_authentication_request(user_id=self.user1['id'], password=self.user1['password']))
        scoped_token = self.get_requested_token(self.build_authentication_request(token=unscoped_token, project_id=self.projectA['id']))
        self.head('/auth/tokens', headers={'X-Subject-Token': unscoped_token}, expected_status=http.client.OK)
        self.head('/auth/tokens', headers={'X-Subject-Token': scoped_token}, expected_status=http.client.OK)
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        self.put('/projects/%(project_id)s/users/%(user_id)s/roles/%(role_id)s' % {'project_id': self.projectA['id'], 'user_id': self.user1['id'], 'role_id': role['id']})
        self.head('/auth/tokens', headers={'X-Subject-Token': unscoped_token}, expected_status=http.client.OK)
        self.head('/auth/tokens', headers={'X-Subject-Token': scoped_token}, expected_status=http.client.OK)

    def test_deleting_user_grant_revokes_token(self):
        """Test deleting a user grant revokes token.

        Test Plan:

        - Get a token for user, scoped to Project
        - Delete the grant user has on Project
        - Check token is no longer valid

        """
        auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=self.project['id'])
        token = self.get_requested_token(auth_data)
        self.head('/auth/tokens', headers={'X-Subject-Token': token}, expected_status=http.client.OK)
        grant_url = '/projects/%(project_id)s/users/%(user_id)s/roles/%(role_id)s' % {'project_id': self.project['id'], 'user_id': self.user['id'], 'role_id': self.role['id']}
        self.delete(grant_url)
        self.head('/auth/tokens', token=token, expected_status=http.client.UNAUTHORIZED)

    def role_data_fixtures(self):
        self.projectC = unit.new_project_ref(domain_id=self.domainA['id'])
        PROVIDERS.resource_api.create_project(self.projectC['id'], self.projectC)
        self.user4 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domainB['id'])
        self.user5 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domainA['id'])
        self.user6 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domainA['id'])
        PROVIDERS.identity_api.add_user_to_group(self.user5['id'], self.group1['id'])
        PROVIDERS.assignment_api.create_grant(self.role1['id'], group_id=self.group1['id'], project_id=self.projectB['id'])
        PROVIDERS.assignment_api.create_grant(self.role2['id'], user_id=self.user4['id'], project_id=self.projectC['id'])
        PROVIDERS.assignment_api.create_grant(self.role1['id'], user_id=self.user6['id'], project_id=self.projectA['id'])
        PROVIDERS.assignment_api.create_grant(self.role1['id'], user_id=self.user6['id'], domain_id=self.domainA['id'])

    def test_deleting_role_revokes_token(self):
        """Test deleting a role revokes token.

        Add some additional test data, namely:

        - A third project (project C)
        - Three additional users - user4 owned by domainB and user5 and 6 owned
          by domainA (different domain ownership should not affect the test
          results, just provided to broaden test coverage)
        - User5 is a member of group1
        - Group1 gets an additional assignment - role1 on projectB as well as
          its existing role1 on projectA
        - User4 has role2 on Project C
        - User6 has role1 on projectA and domainA
        - This allows us to create 5 tokens by virtue of different types of
          role assignment:
          - user1, scoped to ProjectA by virtue of user role1 assignment
          - user5, scoped to ProjectB by virtue of group role1 assignment
          - user4, scoped to ProjectC by virtue of user role2 assignment
          - user6, scoped to ProjectA by virtue of user role1 assignment
          - user6, scoped to DomainA by virtue of user role1 assignment
        - role1 is then deleted
        - Check the tokens on Project A and B, and DomainA are revoked, but not
          the one for Project C

        """
        self.role_data_fixtures()
        auth_data = self.build_authentication_request(user_id=self.user1['id'], password=self.user1['password'], project_id=self.projectA['id'])
        tokenA = self.get_requested_token(auth_data)
        auth_data = self.build_authentication_request(user_id=self.user5['id'], password=self.user5['password'], project_id=self.projectB['id'])
        tokenB = self.get_requested_token(auth_data)
        auth_data = self.build_authentication_request(user_id=self.user4['id'], password=self.user4['password'], project_id=self.projectC['id'])
        tokenC = self.get_requested_token(auth_data)
        auth_data = self.build_authentication_request(user_id=self.user6['id'], password=self.user6['password'], project_id=self.projectA['id'])
        tokenD = self.get_requested_token(auth_data)
        auth_data = self.build_authentication_request(user_id=self.user6['id'], password=self.user6['password'], domain_id=self.domainA['id'])
        tokenE = self.get_requested_token(auth_data)
        self.head('/auth/tokens', headers={'X-Subject-Token': tokenA}, expected_status=http.client.OK)
        self.head('/auth/tokens', headers={'X-Subject-Token': tokenB}, expected_status=http.client.OK)
        self.head('/auth/tokens', headers={'X-Subject-Token': tokenC}, expected_status=http.client.OK)
        self.head('/auth/tokens', headers={'X-Subject-Token': tokenD}, expected_status=http.client.OK)
        self.head('/auth/tokens', headers={'X-Subject-Token': tokenE}, expected_status=http.client.OK)
        role_url = '/roles/%s' % self.role1['id']
        self.delete(role_url)
        self.head('/auth/tokens', headers={'X-Subject-Token': tokenA}, expected_status=http.client.NOT_FOUND)
        self.head('/auth/tokens', headers={'X-Subject-Token': tokenB}, expected_status=http.client.NOT_FOUND)
        self.head('/auth/tokens', headers={'X-Subject-Token': tokenD}, expected_status=http.client.NOT_FOUND)
        self.head('/auth/tokens', headers={'X-Subject-Token': tokenE}, expected_status=http.client.NOT_FOUND)
        self.head('/auth/tokens', headers={'X-Subject-Token': tokenC}, expected_status=http.client.OK)

    def test_domain_user_role_assignment_maintains_token(self):
        """Test user-domain role assignment maintains existing token.

        Test Plan:

        - Get a token for user1, scoped to ProjectA
        - Create a grant for user1 on DomainB
        - Check token is still valid

        """
        auth_data = self.build_authentication_request(user_id=self.user1['id'], password=self.user1['password'], project_id=self.projectA['id'])
        token = self.get_requested_token(auth_data)
        self.head('/auth/tokens', headers={'X-Subject-Token': token}, expected_status=http.client.OK)
        grant_url = '/domains/%(domain_id)s/users/%(user_id)s/roles/%(role_id)s' % {'domain_id': self.domainB['id'], 'user_id': self.user1['id'], 'role_id': self.role1['id']}
        self.put(grant_url)
        self.head('/auth/tokens', headers={'X-Subject-Token': token}, expected_status=http.client.OK)

    def test_disabling_project_revokes_token(self):
        token = self.get_requested_token(self.build_authentication_request(user_id=self.user3['id'], password=self.user3['password'], project_id=self.projectA['id']))
        self.head('/auth/tokens', headers={'X-Subject-Token': token}, expected_status=http.client.OK)
        self.patch('/projects/%(project_id)s' % {'project_id': self.projectA['id']}, body={'project': {'enabled': False}})
        self.head('/auth/tokens', headers={'X-Subject-Token': token}, expected_status=http.client.NOT_FOUND)
        self.v3_create_token(self.build_authentication_request(user_id=self.user3['id'], password=self.user3['password'], project_id=self.projectA['id']), expected_status=http.client.UNAUTHORIZED)

    def test_deleting_project_revokes_token(self):
        token = self.get_requested_token(self.build_authentication_request(user_id=self.user3['id'], password=self.user3['password'], project_id=self.projectA['id']))
        self.head('/auth/tokens', headers={'X-Subject-Token': token}, expected_status=http.client.OK)
        self.delete('/projects/%(project_id)s' % {'project_id': self.projectA['id']})
        self.head('/auth/tokens', headers={'X-Subject-Token': token}, expected_status=http.client.NOT_FOUND)
        self.v3_create_token(self.build_authentication_request(user_id=self.user3['id'], password=self.user3['password'], project_id=self.projectA['id']), expected_status=http.client.UNAUTHORIZED)

    def test_deleting_group_grant_revokes_tokens(self):
        """Test deleting a group grant revokes tokens.

        Test Plan:

        - Get a token for user1, scoped to ProjectA
        - Get a token for user2, scoped to ProjectA
        - Get a token for user3, scoped to ProjectA
        - Delete the grant group1 has on ProjectA
        - Check tokens for user1 & user2 are no longer valid,
          since user1 and user2 are members of group1
        - Check token for user3 is invalid too

        """
        auth_data = self.build_authentication_request(user_id=self.user1['id'], password=self.user1['password'], project_id=self.projectA['id'])
        token1 = self.get_requested_token(auth_data)
        auth_data = self.build_authentication_request(user_id=self.user2['id'], password=self.user2['password'], project_id=self.projectA['id'])
        token2 = self.get_requested_token(auth_data)
        auth_data = self.build_authentication_request(user_id=self.user3['id'], password=self.user3['password'], project_id=self.projectA['id'])
        token3 = self.get_requested_token(auth_data)
        self.head('/auth/tokens', headers={'X-Subject-Token': token1}, expected_status=http.client.OK)
        self.head('/auth/tokens', headers={'X-Subject-Token': token2}, expected_status=http.client.OK)
        self.head('/auth/tokens', headers={'X-Subject-Token': token3}, expected_status=http.client.OK)
        grant_url = '/projects/%(project_id)s/groups/%(group_id)s/roles/%(role_id)s' % {'project_id': self.projectA['id'], 'group_id': self.group1['id'], 'role_id': self.role1['id']}
        self.delete(grant_url)
        PROVIDERS.assignment_api.delete_grant(role_id=self.role1['id'], project_id=self.projectA['id'], user_id=self.user1['id'])
        PROVIDERS.assignment_api.delete_grant(role_id=self.role1['id'], project_id=self.projectA['id'], user_id=self.user2['id'])
        self.head('/auth/tokens', token=token1, expected_status=http.client.UNAUTHORIZED)
        self.head('/auth/tokens', token=token2, expected_status=http.client.UNAUTHORIZED)
        self.head('/auth/tokens', headers={'X-Subject-Token': token3}, expected_status=http.client.OK)

    def test_domain_group_role_assignment_maintains_token(self):
        """Test domain-group role assignment maintains existing token.

        Test Plan:

        - Get a token for user1, scoped to ProjectA
        - Create a grant for group1 on DomainB
        - Check token is still longer valid

        """
        auth_data = self.build_authentication_request(user_id=self.user1['id'], password=self.user1['password'], project_id=self.projectA['id'])
        token = self.get_requested_token(auth_data)
        self.head('/auth/tokens', headers={'X-Subject-Token': token}, expected_status=http.client.OK)
        grant_url = '/domains/%(domain_id)s/groups/%(group_id)s/roles/%(role_id)s' % {'domain_id': self.domainB['id'], 'group_id': self.group1['id'], 'role_id': self.role1['id']}
        self.put(grant_url)
        self.head('/auth/tokens', headers={'X-Subject-Token': token}, expected_status=http.client.OK)

    def test_group_membership_changes_revokes_token(self):
        """Test add/removal to/from group revokes token.

        Test Plan:

        - Get a token for user1, scoped to ProjectA
        - Get a token for user2, scoped to ProjectA
        - Remove user1 from group1
        - Check token for user1 is no longer valid
        - Check token for user2 is still valid, even though
          user2 is also part of group1
        - Add user2 to group2
        - Check token for user2 is now no longer valid

        """
        auth_data = self.build_authentication_request(user_id=self.user1['id'], password=self.user1['password'], project_id=self.projectA['id'])
        token1 = self.get_requested_token(auth_data)
        auth_data = self.build_authentication_request(user_id=self.user2['id'], password=self.user2['password'], project_id=self.projectA['id'])
        token2 = self.get_requested_token(auth_data)
        self.head('/auth/tokens', headers={'X-Subject-Token': token1}, expected_status=http.client.OK)
        self.head('/auth/tokens', headers={'X-Subject-Token': token2}, expected_status=http.client.OK)
        self.delete('/groups/%(group_id)s/users/%(user_id)s' % {'group_id': self.group1['id'], 'user_id': self.user1['id']})
        self.head('/auth/tokens', headers={'X-Subject-Token': token1}, expected_status=http.client.NOT_FOUND)
        self.head('/auth/tokens', headers={'X-Subject-Token': token2}, expected_status=http.client.OK)
        self.put('/groups/%(group_id)s/users/%(user_id)s' % {'group_id': self.group2['id'], 'user_id': self.user2['id']})
        self.head('/auth/tokens', headers={'X-Subject-Token': token2}, expected_status=http.client.OK)

    def test_removing_role_assignment_does_not_affect_other_users(self):
        """Revoking a role from one user should not affect other users."""
        time = datetime.datetime.utcnow()
        with freezegun.freeze_time(time) as frozen_datetime:
            self.delete('/projects/%(p_id)s/groups/%(g_id)s/roles/%(r_id)s' % {'p_id': self.projectA['id'], 'g_id': self.group1['id'], 'r_id': self.role1['id']})
            frozen_datetime.tick(delta=datetime.timedelta(seconds=1))
            user1_token = self.get_requested_token(self.build_authentication_request(user_id=self.user1['id'], password=self.user1['password'], project_id=self.projectA['id']))
            user3_token = self.get_requested_token(self.build_authentication_request(user_id=self.user3['id'], password=self.user3['password'], project_id=self.projectA['id']))
            self.delete('/projects/%(p_id)s/users/%(u_id)s/roles/%(r_id)s' % {'p_id': self.projectA['id'], 'u_id': self.user1['id'], 'r_id': self.role1['id']})
            self.head('/auth/tokens', headers={'X-Subject-Token': user1_token}, expected_status=http.client.NOT_FOUND)
            self.v3_create_token(self.build_authentication_request(user_id=self.user1['id'], password=self.user1['password'], project_id=self.projectA['id']), expected_status=http.client.UNAUTHORIZED)
            self.head('/auth/tokens', headers={'X-Subject-Token': user3_token}, expected_status=http.client.OK)
            self.v3_create_token(self.build_authentication_request(user_id=self.user3['id'], password=self.user3['password'], project_id=self.projectA['id']))

    def test_deleting_project_deletes_grants(self):
        role_path = '/projects/%(project_id)s/users/%(user_id)s/roles/%(role_id)s'
        role_path = role_path % {'user_id': self.user['id'], 'project_id': self.projectA['id'], 'role_id': self.role['id']}
        self.put(role_path)
        self.delete('/projects/%(project_id)s' % {'project_id': self.projectA['id']})
        self.head(role_path, expected_status=http.client.NOT_FOUND)

    def test_revoke_token_from_token(self):
        unscoped_token = self.get_requested_token(self.build_authentication_request(user_id=self.user1['id'], password=self.user1['password']))
        project_scoped_token = self.get_requested_token(self.build_authentication_request(token=unscoped_token, project_id=self.projectA['id']))
        domain_scoped_token = self.get_requested_token(self.build_authentication_request(token=unscoped_token, domain_id=self.domainA['id']))
        self.delete('/auth/tokens', headers={'X-Subject-Token': project_scoped_token})
        self.head('/auth/tokens', headers={'X-Subject-Token': project_scoped_token}, expected_status=http.client.NOT_FOUND)
        self.head('/auth/tokens', headers={'X-Subject-Token': unscoped_token}, expected_status=http.client.OK)
        self.head('/auth/tokens', headers={'X-Subject-Token': domain_scoped_token}, expected_status=http.client.OK)
        self.delete('/auth/tokens', headers={'X-Subject-Token': domain_scoped_token})
        self.head('/auth/tokens', headers={'X-Subject-Token': domain_scoped_token}, expected_status=http.client.NOT_FOUND)
        self.head('/auth/tokens', headers={'X-Subject-Token': unscoped_token}, expected_status=http.client.OK)