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
class GroupSystemRoleAssignmentTestCase(test_v3.RestfulTestCase, SystemRoleAssignmentMixin):

    def test_assign_system_role_to_group(self):
        system_role_id = self._create_new_role()
        group = self._create_group()
        member_url = '/system/groups/%(group_id)s/roles/%(role_id)s' % {'group_id': group['id'], 'role_id': system_role_id}
        self.put(member_url)
        self.head(member_url)
        collection_url = '/system/groups/%(group_id)s/roles' % {'group_id': group['id']}
        roles = self.get(collection_url).json_body['roles']
        self.assertEqual(len(roles), 1)
        self.assertEqual(roles[0]['id'], system_role_id)
        self.head(collection_url, expected_status=http.client.OK)
        response = self.get('/role_assignments?scope.system=all&group.id=%(group_id)s' % {'group_id': group['id']})
        self.assertValidRoleAssignmentListResponse(response, expected_length=1)
        self.assertEqual(response.json_body['role_assignments'][0]['role']['id'], system_role_id)

    def test_assign_system_role_to_non_existant_group_fails(self):
        system_role_id = self._create_new_role()
        group_id = uuid.uuid4().hex
        member_url = '/system/groups/%(group_id)s/roles/%(role_id)s' % {'group_id': group_id, 'role_id': system_role_id}
        self.put(member_url, expected_status=http.client.NOT_FOUND)

    def test_list_role_assignments_for_group_returns_all_assignments(self):
        system_role_id = self._create_new_role()
        group = self._create_group()
        member_url = '/system/groups/%(group_id)s/roles/%(role_id)s' % {'group_id': group['id'], 'role_id': system_role_id}
        self.put(member_url)
        member_url = '/projects/%(project_id)s/groups/%(group_id)s/roles/%(role_id)s' % {'project_id': self.project_id, 'group_id': group['id'], 'role_id': system_role_id}
        self.put(member_url)
        response = self.get('/role_assignments?group.id=%(group_id)s' % {'group_id': group['id']})
        self.assertValidRoleAssignmentListResponse(response, expected_length=2)

    def test_list_system_roles_for_group_returns_none_without_assignment(self):
        group = self._create_group()
        collection_url = '/system/groups/%(group_id)s/roles' % {'group_id': group['id']}
        response = self.get(collection_url)
        self.assertEqual(response.json_body['roles'], [])
        response = self.get('/role_assignments?scope.system=all&group.id=%(group_id)s' % {'group_id': group['id']})
        self.assertValidRoleAssignmentListResponse(response, expected_length=0)

    def test_list_system_roles_for_group_does_not_return_project_roles(self):
        system_role_id = self._create_new_role()
        project_role_id = self._create_new_role()
        group = self._create_group()
        member_url = '/system/groups/%(group_id)s/roles/%(role_id)s' % {'group_id': group['id'], 'role_id': system_role_id}
        self.put(member_url)
        member_url = '/projects/%(project_id)s/groups/%(group_id)s/roles/%(role_id)s' % {'project_id': self.project_id, 'group_id': group['id'], 'role_id': project_role_id}
        self.put(member_url)
        collection_url = '/system/groups/%(group_id)s/roles' % {'group_id': group['id']}
        response = self.get(collection_url)
        for role in response.json_body['roles']:
            self.assertNotEqual(role['id'], project_role_id)
        response = self.get('/role_assignments?scope.system=all&group.id=%(group_id)s' % {'group_id': group['id']})
        self.assertValidRoleAssignmentListResponse(response, expected_length=1)

    def test_list_system_roles_for_group_does_not_return_domain_roles(self):
        system_role_id = self._create_new_role()
        domain_role_id = self._create_new_role()
        group = self._create_group()
        domain_member_url = '/domains/%(domain_id)s/groups/%(group_id)s/roles/%(role_id)s' % {'domain_id': group['domain_id'], 'group_id': group['id'], 'role_id': domain_role_id}
        self.put(domain_member_url)
        member_url = '/system/groups/%(group_id)s/roles/%(role_id)s' % {'group_id': group['id'], 'role_id': system_role_id}
        self.put(member_url)
        response = self.get('/domains/%(domain_id)s/groups/%(group_id)s/roles' % {'domain_id': group['domain_id'], 'group_id': group['id']})
        self.assertEqual(len(response.json_body['roles']), 1)
        collection_url = '/system/groups/%(group_id)s/roles' % {'group_id': group['id']}
        response = self.get(collection_url)
        for role in response.json_body['roles']:
            self.assertNotEqual(role['id'], domain_role_id)
        response = self.get('/role_assignments?scope.system=all&group.id=%(group_id)s' % {'group_id': group['id']})
        self.assertValidRoleAssignmentListResponse(response, expected_length=1)

    def test_check_group_has_system_role_when_assignment_exists(self):
        system_role_id = self._create_new_role()
        group = self._create_group()
        member_url = '/system/groups/%(group_id)s/roles/%(role_id)s' % {'group_id': group['id'], 'role_id': system_role_id}
        self.put(member_url)
        self.head(member_url)
        response = self.get('/role_assignments?scope.system=all&group.id=%(group_id)s' % {'group_id': group['id']})
        self.assertValidRoleAssignmentListResponse(response, expected_length=1)
        self.assertEqual(response.json_body['role_assignments'][0]['role']['id'], system_role_id)

    def test_check_group_does_not_have_system_role_without_assignment(self):
        system_role_id = self._create_new_role()
        group = self._create_group()
        member_url = '/system/groups/%(group_id)s/roles/%(role_id)s' % {'group_id': group['id'], 'role_id': system_role_id}
        self.head(member_url, expected_status=http.client.NOT_FOUND)
        response = self.get('/role_assignments?scope.system=all&group.id=%(group_id)s' % {'group_id': group['id']})
        self.assertValidRoleAssignmentListResponse(response, expected_length=0)

    def test_unassign_system_role_from_group(self):
        system_role_id = self._create_new_role()
        group = self._create_group()
        member_url = '/system/groups/%(group_id)s/roles/%(role_id)s' % {'group_id': group['id'], 'role_id': system_role_id}
        self.put(member_url)
        self.head(member_url)
        response = self.get('/role_assignments?scope.system=all&group.id=%(group_id)s' % {'group_id': group['id']})
        self.assertEqual(len(response.json_body['role_assignments']), 1)
        self.assertValidRoleAssignmentListResponse(response)
        self.delete(member_url)
        collection_url = '/system/groups/%(group_id)s/roles' % {'group_id': group['id']}
        response = self.get(collection_url)
        self.assertEqual(len(response.json_body['roles']), 0)
        response = self.get('/role_assignments?scope.system=all&group.id=%(group_id)s' % {'group_id': group['id']})
        self.assertValidRoleAssignmentListResponse(response, expected_length=0)

    def test_query_for_role_id_does_not_return_system_group_roles(self):
        system_role_id = self._create_new_role()
        group = self._create_group()
        member_url = '/system/groups/%(group_id)s/roles/%(role_id)s' % {'group_id': group['id'], 'role_id': system_role_id}
        self.put(member_url)
        member_url = '/projects/%(project_id)s/groups/%(group_id)s/roles/%(role_id)s' % {'project_id': self.project_id, 'group_id': group['id'], 'role_id': self.role_id}
        self.put(member_url)
        path = '/role_assignments?role.id=%(role_id)s&group.id=%(group_id)s' % {'role_id': self.role_id, 'group_id': group['id']}
        response = self.get(path)
        self.assertValidRoleAssignmentListResponse(response, expected_length=1)