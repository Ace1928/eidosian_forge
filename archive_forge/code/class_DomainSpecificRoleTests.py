import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
class DomainSpecificRoleTests(test_v3.RestfulTestCase, unit.TestCase):

    def setUp(self):

        def create_role(domain_id=None):
            """Call ``POST /roles``."""
            ref = unit.new_role_ref(domain_id=domain_id)
            r = self.post('/roles', body={'role': ref})
            return self.assertValidRoleResponse(r, ref)
        super(DomainSpecificRoleTests, self).setUp()
        self.domainA = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(self.domainA['id'], self.domainA)
        self.domainB = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(self.domainB['id'], self.domainB)
        self.global_role1 = create_role()
        self.global_role2 = create_role()
        r = self.get('/roles')
        self.existing_global_roles = len(r.result['roles'])
        self.domainA_role1 = create_role(domain_id=self.domainA['id'])
        self.domainA_role2 = create_role(domain_id=self.domainA['id'])
        self.domainB_role = create_role(domain_id=self.domainB['id'])

    def test_get_and_list_domain_specific_roles(self):
        r = self.get('/roles/%s' % self.domainA_role1['id'])
        self.assertValidRoleResponse(r, self.domainA_role1)
        r = self.get('/roles')
        self.assertValidRoleListResponse(r, expected_length=self.existing_global_roles)
        self.assertRoleInListResponse(r, self.global_role1)
        self.assertRoleInListResponse(r, self.global_role2)
        self.assertRoleNotInListResponse(r, self.domainA_role1)
        self.assertRoleNotInListResponse(r, self.domainA_role2)
        self.assertRoleNotInListResponse(r, self.domainB_role)
        r = self.get('/roles?domain_id=%s' % self.domainA['id'])
        self.assertValidRoleListResponse(r, expected_length=2)
        self.assertRoleInListResponse(r, self.domainA_role1)
        self.assertRoleInListResponse(r, self.domainA_role2)

    def test_update_domain_specific_roles(self):
        self.domainA_role1['name'] = uuid.uuid4().hex
        self.patch('/roles/%(role_id)s' % {'role_id': self.domainA_role1['id']}, body={'role': self.domainA_role1})
        r = self.get('/roles/%s' % self.domainA_role1['id'])
        self.assertValidRoleResponse(r, self.domainA_role1)

    def test_delete_domain_specific_roles(self):
        self.delete('/roles/%(role_id)s' % {'role_id': self.domainA_role1['id']})
        self.get('/roles/%s' % self.domainA_role1['id'], expected_status=http.client.NOT_FOUND)
        r = self.get('/roles?domain_id=%s' % self.domainA['id'])
        self.assertValidRoleListResponse(r, expected_length=1)
        self.assertRoleInListResponse(r, self.domainA_role2)

    def test_same_domain_assignment(self):
        user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domainA['id'])
        projectA = unit.new_project_ref(domain_id=self.domainA['id'])
        PROVIDERS.resource_api.create_project(projectA['id'], projectA)
        PROVIDERS.assignment_api.create_grant(self.domainA_role1['id'], user_id=user['id'], project_id=projectA['id'])

    def test_cross_domain_assignment_valid(self):
        user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domainB['id'])
        projectA = unit.new_project_ref(domain_id=self.domainA['id'])
        PROVIDERS.resource_api.create_project(projectA['id'], projectA)
        PROVIDERS.assignment_api.create_grant(self.domainA_role1['id'], user_id=user['id'], project_id=projectA['id'])

    def test_cross_domain_assignment_invalid(self):
        user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domainB['id'])
        projectB = unit.new_project_ref(domain_id=self.domainB['id'])
        PROVIDERS.resource_api.create_project(projectB['id'], projectB)
        self.assertRaises(exception.DomainSpecificRoleMismatch, PROVIDERS.assignment_api.create_grant, self.domainA_role1['id'], user_id=user['id'], project_id=projectB['id'])

    def test_cross_domain_implied_roles_authentication(self):
        user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domainB['id'])
        projectA = unit.new_project_ref(domain_id=self.domainA['id'])
        PROVIDERS.resource_api.create_project(projectA['id'], projectA)
        self.put('/roles/%s/implies/%s' % (self.domainA_role1['id'], self.domainB_role['id']), expected_status=http.client.CREATED)
        PROVIDERS.assignment_api.create_grant(self.domainA_role1['id'], user_id=user['id'], project_id=projectA['id'])
        assignments = PROVIDERS.assignment_api.list_role_assignments(user_id=user['id'], effective=True)
        self.assertEqual([], assignments)
        auth_body = self.build_authentication_request(user_id=user['id'], password=user['password'], project_id=projectA['id'])
        self.post('/auth/tokens', body=auth_body, expected_status=http.client.UNAUTHORIZED)