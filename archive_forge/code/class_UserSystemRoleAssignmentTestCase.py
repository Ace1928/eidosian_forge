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
class UserSystemRoleAssignmentTestCase(test_v3.RestfulTestCase, SystemRoleAssignmentMixin):

    def test_assign_system_role_to_user(self):
        system_role_id = self._create_new_role()
        member_url = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': system_role_id}
        self.put(member_url)
        self.head(member_url)
        collection_url = '/system/users/%(user_id)s/roles' % {'user_id': self.user['id']}
        roles = self.get(collection_url).json_body['roles']
        self.assertEqual(len(roles), 1)
        self.assertEqual(roles[0]['id'], system_role_id)
        self.head(collection_url, expected_status=http.client.OK)
        response = self.get('/role_assignments?scope.system=all&user.id=%(user_id)s' % {'user_id': self.user['id']})
        self.assertValidRoleAssignmentListResponse(response)

    def test_list_role_assignments_for_user_returns_all_assignments(self):
        system_role_id = self._create_new_role()
        member_url = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': system_role_id}
        self.put(member_url)
        response = self.get('/role_assignments?user.id=%(user_id)s' % {'user_id': self.user['id']})
        self.assertValidRoleAssignmentListResponse(response, expected_length=2)

    def test_list_system_roles_for_user_returns_none_without_assignment(self):
        collection_url = '/system/users/%(user_id)s/roles' % {'user_id': self.user['id']}
        response = self.get(collection_url)
        self.assertEqual(response.json_body['roles'], [])
        response = self.get('/role_assignments?scope.system=all&user.id=%(user_id)s' % {'user_id': self.user['id']})
        self.assertEqual(len(response.json_body['role_assignments']), 0)
        self.assertValidRoleAssignmentListResponse(response)

    def test_list_system_roles_for_user_does_not_return_project_roles(self):
        system_role_id = self._create_new_role()
        member_url = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': system_role_id}
        self.put(member_url)
        response = self.get('/projects/%(project_id)s/users/%(user_id)s/roles' % {'project_id': self.project['id'], 'user_id': self.user['id']})
        self.assertEqual(len(response.json_body['roles']), 1)
        project_role_id = response.json_body['roles'][0]['id']
        collection_url = '/system/users/%(user_id)s/roles' % {'user_id': self.user['id']}
        response = self.get(collection_url)
        for role in response.json_body['roles']:
            self.assertNotEqual(role['id'], project_role_id)
        response = self.get('/role_assignments?scope.system=all&user.id=%(user_id)s' % {'user_id': self.user['id']})
        self.assertEqual(len(response.json_body['role_assignments']), 1)
        system_assignment = response.json_body['role_assignments'][0]
        self.assertEqual(system_assignment['role']['id'], system_role_id)
        self.assertTrue(system_assignment['scope']['system']['all'])
        path = '/role_assignments?scope.project.id=%(project_id)s&user.id=%(user_id)s' % {'project_id': self.project['id'], 'user_id': self.user['id']}
        response = self.get(path)
        self.assertEqual(len(response.json_body['role_assignments']), 1)
        project_assignment = response.json_body['role_assignments'][0]
        self.assertEqual(project_assignment['role']['id'], project_role_id)

    def test_list_system_roles_for_user_does_not_return_domain_roles(self):
        system_role_id = self._create_new_role()
        domain_role_id = self._create_new_role()
        domain_member_url = '/domains/%(domain_id)s/users/%(user_id)s/roles/%(role_id)s' % {'domain_id': self.user['domain_id'], 'user_id': self.user['id'], 'role_id': domain_role_id}
        self.put(domain_member_url)
        member_url = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': system_role_id}
        self.put(member_url)
        response = self.get('/domains/%(domain_id)s/users/%(user_id)s/roles' % {'domain_id': self.user['domain_id'], 'user_id': self.user['id']})
        self.assertEqual(len(response.json_body['roles']), 1)
        collection_url = '/system/users/%(user_id)s/roles' % {'user_id': self.user['id']}
        response = self.get(collection_url)
        for role in response.json_body['roles']:
            self.assertNotEqual(role['id'], domain_role_id)
        response = self.get('/role_assignments?scope.system=all&user.id=%(user_id)s' % {'user_id': self.user['id']})
        self.assertEqual(len(response.json_body['role_assignments']), 1)
        system_assignment = response.json_body['role_assignments'][0]
        self.assertEqual(system_assignment['role']['id'], system_role_id)
        self.assertTrue(system_assignment['scope']['system']['all'])
        path = '/role_assignments?scope.domain.id=%(domain_id)s&user.id=%(user_id)s' % {'domain_id': self.user['domain_id'], 'user_id': self.user['id']}
        response = self.get(path)
        self.assertEqual(len(response.json_body['role_assignments']), 1)
        domain_assignment = response.json_body['role_assignments'][0]
        self.assertEqual(domain_assignment['role']['id'], domain_role_id)

    def test_check_user_has_system_role_when_assignment_exists(self):
        system_role_id = self._create_new_role()
        member_url = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': system_role_id}
        self.put(member_url)
        self.head(member_url)

    def test_check_user_does_not_have_system_role_without_assignment(self):
        system_role_id = self._create_new_role()
        member_url = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': system_role_id}
        self.head(member_url, expected_status=http.client.NOT_FOUND)
        response = self.get('/role_assignments?scope.system=all&user.id=%(user_id)s' % {'user_id': self.user['id']})
        self.assertEqual(len(response.json_body['role_assignments']), 0)
        self.assertValidRoleAssignmentListResponse(response)

    def test_unassign_system_role_from_user(self):
        system_role_id = self._create_new_role()
        member_url = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': system_role_id}
        self.put(member_url)
        self.head(member_url)
        response = self.get('/role_assignments?scope.system=all&user.id=%(user_id)s' % {'user_id': self.user['id']})
        self.assertEqual(len(response.json_body['role_assignments']), 1)
        self.assertValidRoleAssignmentListResponse(response)
        self.delete(member_url)
        collection_url = '/system/users/%(user_id)s/roles' % {'user_id': self.user['id']}
        response = self.get(collection_url)
        self.assertEqual(len(response.json_body['roles']), 0)
        response = self.get('/role_assignments?scope.system=all&user.id=%(user_id)s' % {'user_id': self.user['id']})
        self.assertValidRoleAssignmentListResponse(response, expected_length=0)

    def test_query_for_system_scope_and_domain_scope_fails(self):
        path = '/role_assignments?scope.system=all&scope.domain.id=%(domain_id)s' % {'domain_id': self.domain_id}
        self.get(path, expected_status=http.client.BAD_REQUEST)

    def test_query_for_system_scope_and_project_scope_fails(self):
        path = '/role_assignments?scope.system=all&scope.project.id=%(project_id)s' % {'project_id': self.project_id}
        self.get(path, expected_status=http.client.BAD_REQUEST)

    def test_query_for_role_id_does_not_return_system_user_roles(self):
        system_role_id = self._create_new_role()
        member_url = '/system/users/%(user_id)s/roles/%(role_id)s' % {'user_id': self.user['id'], 'role_id': system_role_id}
        self.put(member_url)
        path = '/role_assignments?role.id=%(role_id)s&user.id=%(user_id)s' % {'role_id': self.role_id, 'user_id': self.user['id']}
        response = self.get(path)
        self.assertValidRoleAssignmentListResponse(response, expected_length=1)