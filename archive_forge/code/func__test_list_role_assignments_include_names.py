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
def _test_list_role_assignments_include_names(self, role1):
    """Call ``GET /role_assignments with include names``.

        Test Plan:

        - Create a domain with a group and a user
        - Create a project with a group and a user

        """
    role1 = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role1['id'], role1)
    user1 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
    group = unit.new_group_ref(domain_id=self.domain_id)
    group = PROVIDERS.identity_api.create_group(group)
    project1 = unit.new_project_ref(domain_id=self.domain_id)
    PROVIDERS.resource_api.create_project(project1['id'], project1)
    expected_entity1 = self.build_role_assignment_entity_include_names(role_ref=role1, project_ref=project1, user_ref=user1)
    self.put(expected_entity1['links']['assignment'])
    expected_entity2 = self.build_role_assignment_entity_include_names(role_ref=role1, domain_ref=self.domain, group_ref=group)
    self.put(expected_entity2['links']['assignment'])
    expected_entity3 = self.build_role_assignment_entity_include_names(role_ref=role1, domain_ref=self.domain, user_ref=user1)
    self.put(expected_entity3['links']['assignment'])
    expected_entity4 = self.build_role_assignment_entity_include_names(role_ref=role1, project_ref=project1, group_ref=group)
    self.put(expected_entity4['links']['assignment'])
    collection_url_domain = '/role_assignments?include_names&scope.domain.id=%(domain_id)s' % {'domain_id': self.domain_id}
    rs_domain = self.get(collection_url_domain)
    collection_url_project = '/role_assignments?include_names&scope.project.id=%(project_id)s' % {'project_id': project1['id']}
    rs_project = self.get(collection_url_project)
    collection_url_group = '/role_assignments?include_names&group.id=%(group_id)s' % {'group_id': group['id']}
    rs_group = self.get(collection_url_group)
    collection_url_user = '/role_assignments?include_names&user.id=%(user_id)s' % {'user_id': user1['id']}
    rs_user = self.get(collection_url_user)
    collection_url_role = '/role_assignments?include_names&role.id=%(role_id)s' % {'role_id': role1['id']}
    rs_role = self.get(collection_url_role)
    self.assertEqual(http.client.OK, rs_domain.status_int)
    self.assertEqual(http.client.OK, rs_project.status_int)
    self.assertEqual(http.client.OK, rs_group.status_int)
    self.assertEqual(http.client.OK, rs_user.status_int)
    self.assertValidRoleAssignmentListResponse(rs_domain, expected_length=2, resource_url=collection_url_domain)
    self.assertValidRoleAssignmentListResponse(rs_project, expected_length=2, resource_url=collection_url_project)
    self.assertValidRoleAssignmentListResponse(rs_group, expected_length=2, resource_url=collection_url_group)
    self.assertValidRoleAssignmentListResponse(rs_user, expected_length=2, resource_url=collection_url_user)
    self.assertValidRoleAssignmentListResponse(rs_role, expected_length=4, resource_url=collection_url_role)
    self.assertRoleAssignmentInListResponse(rs_domain, expected_entity2)
    self.assertRoleAssignmentInListResponse(rs_project, expected_entity1)
    self.assertRoleAssignmentInListResponse(rs_group, expected_entity4)
    self.assertRoleAssignmentInListResponse(rs_user, expected_entity3)
    self.assertRoleAssignmentInListResponse(rs_role, expected_entity1)