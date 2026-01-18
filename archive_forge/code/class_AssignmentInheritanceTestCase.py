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
class AssignmentInheritanceTestCase(test_v3.RestfulTestCase, test_v3.AssignmentTestMixin):
    """Test inheritance crud and its effects."""

    def test_get_token_from_inherited_user_domain_role_grants(self):
        user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
        domain_auth_data = self.build_authentication_request(user_id=user['id'], password=user['password'], domain_id=self.domain_id)
        project_auth_data = self.build_authentication_request(user_id=user['id'], password=user['password'], project_id=self.project_id)
        self.v3_create_token(domain_auth_data, expected_status=http.client.UNAUTHORIZED)
        self.v3_create_token(project_auth_data, expected_status=http.client.UNAUTHORIZED)
        non_inher_ud_link = self.build_role_assignment_link(domain_id=self.domain_id, user_id=user['id'], role_id=self.role_id)
        self.put(non_inher_ud_link)
        self.v3_create_token(domain_auth_data)
        self.v3_create_token(project_auth_data, expected_status=http.client.UNAUTHORIZED)
        inherited_role = unit.new_role_ref(name='inherited')
        PROVIDERS.role_api.create_role(inherited_role['id'], inherited_role)
        inher_ud_link = self.build_role_assignment_link(domain_id=self.domain_id, user_id=user['id'], role_id=inherited_role['id'], inherited_to_projects=True)
        self.put(inher_ud_link)
        self.v3_create_token(domain_auth_data)
        self.v3_create_token(project_auth_data)
        self.delete(inher_ud_link)
        self.v3_create_token(domain_auth_data)
        self.v3_create_token(project_auth_data, expected_status=http.client.UNAUTHORIZED)
        self.delete(non_inher_ud_link)
        self.v3_create_token(domain_auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_get_token_from_inherited_group_domain_role_grants(self):
        user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
        group = unit.new_group_ref(domain_id=self.domain['id'])
        group = PROVIDERS.identity_api.create_group(group)
        PROVIDERS.identity_api.add_user_to_group(user['id'], group['id'])
        domain_auth_data = self.build_authentication_request(user_id=user['id'], password=user['password'], domain_id=self.domain_id)
        project_auth_data = self.build_authentication_request(user_id=user['id'], password=user['password'], project_id=self.project_id)
        self.v3_create_token(domain_auth_data, expected_status=http.client.UNAUTHORIZED)
        self.v3_create_token(project_auth_data, expected_status=http.client.UNAUTHORIZED)
        non_inher_gd_link = self.build_role_assignment_link(domain_id=self.domain_id, user_id=user['id'], role_id=self.role_id)
        self.put(non_inher_gd_link)
        self.v3_create_token(domain_auth_data)
        self.v3_create_token(project_auth_data, expected_status=http.client.UNAUTHORIZED)
        inherited_role = unit.new_role_ref(name='inherited')
        PROVIDERS.role_api.create_role(inherited_role['id'], inherited_role)
        inher_gd_link = self.build_role_assignment_link(domain_id=self.domain_id, user_id=user['id'], role_id=inherited_role['id'], inherited_to_projects=True)
        self.put(inher_gd_link)
        self.v3_create_token(domain_auth_data)
        self.v3_create_token(project_auth_data)
        self.delete(inher_gd_link)
        self.v3_create_token(domain_auth_data)
        self.v3_create_token(project_auth_data, expected_status=http.client.UNAUTHORIZED)
        self.delete(non_inher_gd_link)
        self.v3_create_token(domain_auth_data, expected_status=http.client.UNAUTHORIZED)

    def _test_crud_inherited_and_direct_assignment_on_target(self, target_url):
        time = datetime.datetime.utcnow()
        with freezegun.freeze_time(time) as frozen_datetime:
            role = unit.new_role_ref()
            PROVIDERS.role_api.create_role(role['id'], role)
            direct_url = '%s/users/%s/roles/%s' % (target_url, self.user_id, role['id'])
            inherited_url = '/OS-INHERIT/%s/inherited_to_projects' % direct_url.lstrip('/')
            self.put(direct_url)
            self.head(direct_url)
            self.head(inherited_url, expected_status=http.client.NOT_FOUND)
            self.put(inherited_url)
            self.head(direct_url)
            self.head(inherited_url)
            self.delete(inherited_url)
            frozen_datetime.tick(delta=datetime.timedelta(seconds=1))
            self.head(direct_url)
            self.head(inherited_url, expected_status=http.client.NOT_FOUND)
            self.delete(direct_url)
            self.head(direct_url, expected_status=http.client.NOT_FOUND)
            self.head(inherited_url, expected_status=http.client.NOT_FOUND)

    def test_crud_inherited_and_direct_assignment_on_domains(self):
        self._test_crud_inherited_and_direct_assignment_on_target('/domains/%s' % self.domain_id)

    def test_crud_inherited_and_direct_assignment_on_projects(self):
        self._test_crud_inherited_and_direct_assignment_on_target('/projects/%s' % self.project_id)

    def test_crud_user_inherited_domain_role_grants(self):
        role_list = []
        for _ in range(2):
            role = unit.new_role_ref()
            PROVIDERS.role_api.create_role(role['id'], role)
            role_list.append(role)
        PROVIDERS.assignment_api.create_grant(role_list[1]['id'], user_id=self.user['id'], domain_id=self.domain_id)
        base_collection_url = '/OS-INHERIT/domains/%(domain_id)s/users/%(user_id)s/roles' % {'domain_id': self.domain_id, 'user_id': self.user['id']}
        member_url = '%(collection_url)s/%(role_id)s/inherited_to_projects' % {'collection_url': base_collection_url, 'role_id': role_list[0]['id']}
        collection_url = base_collection_url + '/inherited_to_projects'
        self.put(member_url)
        self.head(member_url)
        self.get(member_url, expected_status=http.client.NO_CONTENT)
        r = self.get(collection_url)
        self.assertValidRoleListResponse(r, ref=role_list[0], resource_url=collection_url)
        self.delete(member_url)
        r = self.get(collection_url)
        self.assertValidRoleListResponse(r, expected_length=0, resource_url=collection_url)

    def test_list_role_assignments_for_inherited_domain_grants(self):
        """Call ``GET /role_assignments with inherited domain grants``.

        Test Plan:

        - Create 4 roles
        - Create a domain with a user and two projects
        - Assign two direct roles to project1
        - Assign a spoiler role to project2
        - Issue the URL to add inherited role to the domain
        - Issue the URL to check it is indeed on the domain
        - Issue the URL to check effective roles on project1 - this
          should return 3 roles.

        """
        role_list = []
        for _ in range(4):
            role = unit.new_role_ref()
            PROVIDERS.role_api.create_role(role['id'], role)
            role_list.append(role)
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        user1 = unit.create_user(PROVIDERS.identity_api, domain_id=domain['id'])
        project1 = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project1['id'], project1)
        project2 = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project2['id'], project2)
        PROVIDERS.assignment_api.add_role_to_user_and_project(user1['id'], project1['id'], role_list[0]['id'])
        PROVIDERS.assignment_api.add_role_to_user_and_project(user1['id'], project1['id'], role_list[1]['id'])
        PROVIDERS.assignment_api.add_role_to_user_and_project(user1['id'], project2['id'], role_list[2]['id'])
        base_collection_url = '/OS-INHERIT/domains/%(domain_id)s/users/%(user_id)s/roles' % {'domain_id': domain['id'], 'user_id': user1['id']}
        member_url = '%(collection_url)s/%(role_id)s/inherited_to_projects' % {'collection_url': base_collection_url, 'role_id': role_list[3]['id']}
        collection_url = base_collection_url + '/inherited_to_projects'
        self.put(member_url)
        self.head(member_url)
        self.get(member_url, expected_status=http.client.NO_CONTENT)
        r = self.get(collection_url)
        self.assertValidRoleListResponse(r, ref=role_list[3], resource_url=collection_url)
        collection_url = '/role_assignments?user.id=%(user_id)s&scope.domain.id=%(domain_id)s' % {'user_id': user1['id'], 'domain_id': domain['id']}
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, expected_length=1, resource_url=collection_url)
        ud_entity = self.build_role_assignment_entity(domain_id=domain['id'], user_id=user1['id'], role_id=role_list[3]['id'], inherited_to_projects=True)
        self.assertRoleAssignmentInListResponse(r, ud_entity)
        collection_url = '/role_assignments?effective&user.id=%(user_id)s&scope.project.id=%(project_id)s' % {'user_id': user1['id'], 'project_id': project1['id']}
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, expected_length=3, resource_url=collection_url)
        ud_url = self.build_role_assignment_link(domain_id=domain['id'], user_id=user1['id'], role_id=role_list[3]['id'], inherited_to_projects=True)
        up_entity = self.build_role_assignment_entity(link=ud_url, project_id=project1['id'], user_id=user1['id'], role_id=role_list[3]['id'], inherited_to_projects=True)
        self.assertRoleAssignmentInListResponse(r, up_entity)

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

    def test_list_role_assignments_include_names_global_role(self):
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        self._test_list_role_assignments_include_names(role)

    def test_list_role_assignments_include_names_domain_role(self):
        role = unit.new_role_ref(domain_id=self.domain['id'])
        PROVIDERS.role_api.create_role(role['id'], role)
        self._test_list_role_assignments_include_names(role)

    def test_remove_assignment_for_project_acting_as_domain(self):
        """Test goal: remove assignment for project acting as domain.

        Ensure when we have two role assignments for the project
        acting as domain, one dealing with it as a domain and other as a
        project, we still able to remove those assignments later.

        Test plan:
        - Create a role and a domain with a user;
        - Grant a role for this user in this domain;
        - Grant a role for this user in the same entity as a project;
        - Ensure that both assignments were created and it was valid;
        - Remove the domain assignment for the user and show that the project
        assignment for him still valid

        """
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        user = unit.create_user(PROVIDERS.identity_api, domain_id=domain['id'])
        assignment_domain = self.build_role_assignment_entity(role_id=role['id'], domain_id=domain['id'], user_id=user['id'], inherited_to_projects=False)
        assignment_project = self.build_role_assignment_entity(role_id=role['id'], project_id=domain['id'], user_id=user['id'], inherited_to_projects=False)
        self.put(assignment_domain['links']['assignment'])
        self.put(assignment_project['links']['assignment'])
        collection_url = '/role_assignments?user.id=%(user_id)s' % {'user_id': user['id']}
        result = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(result, expected_length=2, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(result, assignment_domain)
        domain_url = '/domains/%s/users/%s/roles/%s' % (domain['id'], user['id'], role['id'])
        self.delete(domain_url)
        collection_url = '/role_assignments?user.id=%(user_id)s' % {'user_id': user['id']}
        result = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(result, expected_length=1, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(result, assignment_project)

    def test_list_inherited_role_assignments_include_names(self):
        """Call ``GET /role_assignments?include_names``.

        Test goal: ensure calling list role assignments including names
        honors the inherited role assignments flag.

        Test plan:
        - Create a role and a domain with a user;
        - Create a inherited role assignment;
        - List role assignments for that user;
        - List role assignments for that user including names.

        """
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        user = unit.create_user(PROVIDERS.identity_api, domain_id=domain['id'])
        assignment = self.build_role_assignment_entity(role_id=role['id'], domain_id=domain['id'], user_id=user['id'], inherited_to_projects=True)
        assignment_names = self.build_role_assignment_entity_include_names(role_ref=role, domain_ref=domain, user_ref=user, inherited_assignment=True)
        self.assertEqual('projects', assignment['scope']['OS-INHERIT:inherited_to'])
        self.assertEqual('projects', assignment_names['scope']['OS-INHERIT:inherited_to'])
        self.assertEqual(assignment['links']['assignment'], assignment_names['links']['assignment'])
        self.put(assignment['links']['assignment'])
        collection_url = '/role_assignments?user.id=%(user_id)s' % {'user_id': user['id']}
        result = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(result, expected_length=1, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(result, assignment)
        collection_url = '/role_assignments?include_names&user.id=%(user_id)s' % {'user_id': user['id']}
        result = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(result, expected_length=1, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(result, assignment_names)

    def test_list_role_assignments_for_disabled_inheritance_extension(self):
        """Call ``GET /role_assignments with inherited domain grants``.

        Test Plan:

        - Issue the URL to add inherited role to the domain
        - Issue the URL to check effective roles on project include the
          inherited role
        - Disable the extension
        - Re-check the effective roles, proving the inherited role no longer
          shows up.

        """
        role_list = []
        for _ in range(4):
            role = unit.new_role_ref()
            PROVIDERS.role_api.create_role(role['id'], role)
            role_list.append(role)
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        user1 = unit.create_user(PROVIDERS.identity_api, domain_id=domain['id'])
        project1 = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project1['id'], project1)
        project2 = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project2['id'], project2)
        PROVIDERS.assignment_api.add_role_to_user_and_project(user1['id'], project1['id'], role_list[0]['id'])
        PROVIDERS.assignment_api.add_role_to_user_and_project(user1['id'], project1['id'], role_list[1]['id'])
        PROVIDERS.assignment_api.add_role_to_user_and_project(user1['id'], project2['id'], role_list[2]['id'])
        base_collection_url = '/OS-INHERIT/domains/%(domain_id)s/users/%(user_id)s/roles' % {'domain_id': domain['id'], 'user_id': user1['id']}
        member_url = '%(collection_url)s/%(role_id)s/inherited_to_projects' % {'collection_url': base_collection_url, 'role_id': role_list[3]['id']}
        collection_url = base_collection_url + '/inherited_to_projects'
        self.put(member_url)
        self.head(member_url)
        self.get(member_url, expected_status=http.client.NO_CONTENT)
        r = self.get(collection_url)
        self.assertValidRoleListResponse(r, ref=role_list[3], resource_url=collection_url)
        collection_url = '/role_assignments?effective&user.id=%(user_id)s&scope.project.id=%(project_id)s' % {'user_id': user1['id'], 'project_id': project1['id']}
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, expected_length=3, resource_url=collection_url)
        ud_url = self.build_role_assignment_link(domain_id=domain['id'], user_id=user1['id'], role_id=role_list[3]['id'], inherited_to_projects=True)
        up_entity = self.build_role_assignment_entity(link=ud_url, project_id=project1['id'], user_id=user1['id'], role_id=role_list[3]['id'], inherited_to_projects=True)
        self.assertRoleAssignmentInListResponse(r, up_entity)

    def test_list_role_assignments_for_inherited_group_domain_grants(self):
        """Call ``GET /role_assignments with inherited group domain grants``.

        Test Plan:

        - Create 4 roles
        - Create a domain with a user and two projects
        - Assign two direct roles to project1
        - Assign a spoiler role to project2
        - Issue the URL to add inherited role to the domain
        - Issue the URL to check it is indeed on the domain
        - Issue the URL to check effective roles on project1 - this
          should return 3 roles.

        """
        role_list = []
        for _ in range(4):
            role = unit.new_role_ref()
            PROVIDERS.role_api.create_role(role['id'], role)
            role_list.append(role)
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        user1 = unit.create_user(PROVIDERS.identity_api, domain_id=domain['id'])
        user2 = unit.create_user(PROVIDERS.identity_api, domain_id=domain['id'])
        group1 = unit.new_group_ref(domain_id=domain['id'])
        group1 = PROVIDERS.identity_api.create_group(group1)
        PROVIDERS.identity_api.add_user_to_group(user1['id'], group1['id'])
        PROVIDERS.identity_api.add_user_to_group(user2['id'], group1['id'])
        project1 = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project1['id'], project1)
        project2 = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project2['id'], project2)
        PROVIDERS.assignment_api.add_role_to_user_and_project(user1['id'], project1['id'], role_list[0]['id'])
        PROVIDERS.assignment_api.add_role_to_user_and_project(user1['id'], project1['id'], role_list[1]['id'])
        PROVIDERS.assignment_api.add_role_to_user_and_project(user1['id'], project2['id'], role_list[2]['id'])
        base_collection_url = '/OS-INHERIT/domains/%(domain_id)s/groups/%(group_id)s/roles' % {'domain_id': domain['id'], 'group_id': group1['id']}
        member_url = '%(collection_url)s/%(role_id)s/inherited_to_projects' % {'collection_url': base_collection_url, 'role_id': role_list[3]['id']}
        collection_url = base_collection_url + '/inherited_to_projects'
        self.put(member_url)
        self.head(member_url)
        self.get(member_url, expected_status=http.client.NO_CONTENT)
        r = self.get(collection_url)
        self.assertValidRoleListResponse(r, ref=role_list[3], resource_url=collection_url)
        collection_url = '/role_assignments?group.id=%(group_id)s&scope.domain.id=%(domain_id)s' % {'group_id': group1['id'], 'domain_id': domain['id']}
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, expected_length=1, resource_url=collection_url)
        gd_entity = self.build_role_assignment_entity(domain_id=domain['id'], group_id=group1['id'], role_id=role_list[3]['id'], inherited_to_projects=True)
        self.assertRoleAssignmentInListResponse(r, gd_entity)
        collection_url = '/role_assignments?effective&user.id=%(user_id)s&scope.project.id=%(project_id)s' % {'user_id': user1['id'], 'project_id': project1['id']}
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, expected_length=3, resource_url=collection_url)
        up_entity = self.build_role_assignment_entity(link=gd_entity['links']['assignment'], project_id=project1['id'], user_id=user1['id'], role_id=role_list[3]['id'], inherited_to_projects=True)
        self.assertRoleAssignmentInListResponse(r, up_entity)

    def test_filtered_role_assignments_for_inherited_grants(self):
        """Call ``GET /role_assignments?scope.OS-INHERIT:inherited_to``.

        Test Plan:

        - Create 5 roles
        - Create a domain with a user, group and two projects
        - Assign three direct spoiler roles to projects
        - Issue the URL to add an inherited user role to the domain
        - Issue the URL to add an inherited group role to the domain
        - Issue the URL to filter by inherited roles - this should
          return just the 2 inherited roles.

        """
        role_list = []
        for _ in range(5):
            role = unit.new_role_ref()
            PROVIDERS.role_api.create_role(role['id'], role)
            role_list.append(role)
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        user1 = unit.create_user(PROVIDERS.identity_api, domain_id=domain['id'])
        group1 = unit.new_group_ref(domain_id=domain['id'])
        group1 = PROVIDERS.identity_api.create_group(group1)
        project1 = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project1['id'], project1)
        project2 = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project2['id'], project2)
        PROVIDERS.assignment_api.add_role_to_user_and_project(user1['id'], project1['id'], role_list[0]['id'])
        PROVIDERS.assignment_api.add_role_to_user_and_project(user1['id'], project2['id'], role_list[1]['id'])
        PROVIDERS.assignment_api.create_grant(role_list[2]['id'], user_id=user1['id'], domain_id=domain['id'])
        base_collection_url = '/OS-INHERIT/domains/%(domain_id)s/users/%(user_id)s/roles' % {'domain_id': domain['id'], 'user_id': user1['id']}
        member_url = '%(collection_url)s/%(role_id)s/inherited_to_projects' % {'collection_url': base_collection_url, 'role_id': role_list[3]['id']}
        collection_url = base_collection_url + '/inherited_to_projects'
        self.put(member_url)
        self.head(member_url)
        self.get(member_url, expected_status=http.client.NO_CONTENT)
        r = self.get(collection_url)
        self.assertValidRoleListResponse(r, ref=role_list[3], resource_url=collection_url)
        base_collection_url = '/OS-INHERIT/domains/%(domain_id)s/groups/%(group_id)s/roles' % {'domain_id': domain['id'], 'group_id': group1['id']}
        member_url = '%(collection_url)s/%(role_id)s/inherited_to_projects' % {'collection_url': base_collection_url, 'role_id': role_list[4]['id']}
        collection_url = base_collection_url + '/inherited_to_projects'
        self.put(member_url)
        self.head(member_url)
        self.get(member_url, expected_status=http.client.NO_CONTENT)
        r = self.get(collection_url)
        self.assertValidRoleListResponse(r, ref=role_list[4], resource_url=collection_url)
        collection_url = '/role_assignments?scope.OS-INHERIT:inherited_to=projects'
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, expected_length=2, resource_url=collection_url)
        ud_entity = self.build_role_assignment_entity(domain_id=domain['id'], user_id=user1['id'], role_id=role_list[3]['id'], inherited_to_projects=True)
        gd_entity = self.build_role_assignment_entity(domain_id=domain['id'], group_id=group1['id'], role_id=role_list[4]['id'], inherited_to_projects=True)
        self.assertRoleAssignmentInListResponse(r, ud_entity)
        self.assertRoleAssignmentInListResponse(r, gd_entity)

    def _setup_hierarchical_projects_scenario(self):
        """Create basic hierarchical projects scenario.

        This basic scenario contains a root with one leaf project and
        two roles with the following names: non-inherited and inherited.

        """
        root = unit.new_project_ref(domain_id=self.domain['id'])
        leaf = unit.new_project_ref(domain_id=self.domain['id'], parent_id=root['id'])
        PROVIDERS.resource_api.create_project(root['id'], root)
        PROVIDERS.resource_api.create_project(leaf['id'], leaf)
        non_inherited_role = unit.new_role_ref(name='non-inherited')
        PROVIDERS.role_api.create_role(non_inherited_role['id'], non_inherited_role)
        inherited_role = unit.new_role_ref(name='inherited')
        PROVIDERS.role_api.create_role(inherited_role['id'], inherited_role)
        return (root['id'], leaf['id'], non_inherited_role['id'], inherited_role['id'])

    def test_get_token_from_inherited_user_project_role_grants(self):
        root_id, leaf_id, non_inherited_role_id, inherited_role_id = self._setup_hierarchical_projects_scenario()
        root_project_auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=root_id)
        leaf_project_auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=leaf_id)
        self.v3_create_token(root_project_auth_data, expected_status=http.client.UNAUTHORIZED)
        self.v3_create_token(leaf_project_auth_data, expected_status=http.client.UNAUTHORIZED)
        non_inher_up_link = self.build_role_assignment_link(project_id=leaf_id, user_id=self.user['id'], role_id=non_inherited_role_id)
        self.put(non_inher_up_link)
        self.v3_create_token(root_project_auth_data, expected_status=http.client.UNAUTHORIZED)
        self.v3_create_token(leaf_project_auth_data)
        inher_up_link = self.build_role_assignment_link(project_id=root_id, user_id=self.user['id'], role_id=inherited_role_id, inherited_to_projects=True)
        self.put(inher_up_link)
        self.v3_create_token(root_project_auth_data, expected_status=http.client.UNAUTHORIZED)
        self.v3_create_token(leaf_project_auth_data)
        self.delete(non_inher_up_link)
        self.v3_create_token(root_project_auth_data, expected_status=http.client.UNAUTHORIZED)
        self.v3_create_token(leaf_project_auth_data)
        self.delete(inher_up_link)
        self.v3_create_token(leaf_project_auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_get_token_from_inherited_group_project_role_grants(self):
        root_id, leaf_id, non_inherited_role_id, inherited_role_id = self._setup_hierarchical_projects_scenario()
        group = unit.new_group_ref(domain_id=self.domain['id'])
        group = PROVIDERS.identity_api.create_group(group)
        PROVIDERS.identity_api.add_user_to_group(self.user['id'], group['id'])
        root_project_auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=root_id)
        leaf_project_auth_data = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], project_id=leaf_id)
        self.v3_create_token(root_project_auth_data, expected_status=http.client.UNAUTHORIZED)
        self.v3_create_token(leaf_project_auth_data, expected_status=http.client.UNAUTHORIZED)
        non_inher_gp_link = self.build_role_assignment_link(project_id=leaf_id, group_id=group['id'], role_id=non_inherited_role_id)
        self.put(non_inher_gp_link)
        self.v3_create_token(root_project_auth_data, expected_status=http.client.UNAUTHORIZED)
        self.v3_create_token(leaf_project_auth_data)
        inher_gp_link = self.build_role_assignment_link(project_id=root_id, group_id=group['id'], role_id=inherited_role_id, inherited_to_projects=True)
        self.put(inher_gp_link)
        self.v3_create_token(root_project_auth_data, expected_status=http.client.UNAUTHORIZED)
        self.v3_create_token(leaf_project_auth_data)
        self.delete(non_inher_gp_link)
        self.v3_create_token(leaf_project_auth_data)
        self.delete(inher_gp_link)
        self.v3_create_token(leaf_project_auth_data, expected_status=http.client.UNAUTHORIZED)

    def test_get_role_assignments_for_project_hierarchy(self):
        """Call ``GET /role_assignments``.

        Test Plan:

        - Create 2 roles
        - Create a hierarchy of projects with one root and one leaf project
        - Issue the URL to add a non-inherited user role to the root project
        - Issue the URL to add an inherited user role to the root project
        - Issue the URL to get all role assignments - this should return just
          2 roles (non-inherited and inherited) in the root project.

        """
        root_id, leaf_id, non_inherited_role_id, inherited_role_id = self._setup_hierarchical_projects_scenario()
        non_inher_up_entity = self.build_role_assignment_entity(project_id=root_id, user_id=self.user['id'], role_id=non_inherited_role_id)
        self.put(non_inher_up_entity['links']['assignment'])
        inher_up_entity = self.build_role_assignment_entity(project_id=root_id, user_id=self.user['id'], role_id=inherited_role_id, inherited_to_projects=True)
        self.put(inher_up_entity['links']['assignment'])
        collection_url = '/role_assignments'
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(r, non_inher_up_entity)
        self.assertRoleAssignmentInListResponse(r, inher_up_entity)
        non_inher_up_entity = self.build_role_assignment_entity(project_id=leaf_id, user_id=self.user['id'], role_id=non_inherited_role_id)
        self.assertRoleAssignmentNotInListResponse(r, non_inher_up_entity)
        inher_up_entity['scope']['project']['id'] = leaf_id
        self.assertRoleAssignmentNotInListResponse(r, inher_up_entity)

    def test_get_effective_role_assignments_for_project_hierarchy(self):
        """Call ``GET /role_assignments?effective``.

        Test Plan:

        - Create 2 roles
        - Create a hierarchy of projects with one root and one leaf project
        - Issue the URL to add a non-inherited user role to the root project
        - Issue the URL to add an inherited user role to the root project
        - Issue the URL to get effective role assignments - this should return
          1 role (non-inherited) on the root project and 1 role (inherited) on
          the leaf project.

        """
        root_id, leaf_id, non_inherited_role_id, inherited_role_id = self._setup_hierarchical_projects_scenario()
        non_inher_up_entity = self.build_role_assignment_entity(project_id=root_id, user_id=self.user['id'], role_id=non_inherited_role_id)
        self.put(non_inher_up_entity['links']['assignment'])
        inher_up_entity = self.build_role_assignment_entity(project_id=root_id, user_id=self.user['id'], role_id=inherited_role_id, inherited_to_projects=True)
        self.put(inher_up_entity['links']['assignment'])
        collection_url = '/role_assignments?effective'
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, resource_url=collection_url)
        self.assertRoleAssignmentInListResponse(r, non_inher_up_entity)
        self.assertRoleAssignmentNotInListResponse(r, inher_up_entity)
        non_inher_up_entity = self.build_role_assignment_entity(project_id=leaf_id, user_id=self.user['id'], role_id=non_inherited_role_id)
        self.assertRoleAssignmentNotInListResponse(r, non_inher_up_entity)
        inher_up_entity['scope']['project']['id'] = leaf_id
        self.assertRoleAssignmentInListResponse(r, inher_up_entity)

    def test_project_id_specified_if_include_subtree_specified(self):
        """When using include_subtree, you must specify a project ID."""
        r = self.get('/role_assignments?include_subtree=True', expected_status=http.client.BAD_REQUEST)
        error_msg = 'scope.project.id must be specified if include_subtree is also specified'
        self.assertEqual(error_msg, r.result['error']['message'])
        r = self.get('/role_assignments?scope.project.id&include_subtree=True', expected_status=http.client.BAD_REQUEST)
        self.assertEqual(error_msg, r.result['error']['message'])

    def test_get_role_assignments_for_project_tree(self):
        """Get role_assignment?scope.project.id=X&include_subtree``.

        Test Plan:

        - Create 2 roles and a hierarchy of projects with one root and one leaf
        - Issue the URL to add a non-inherited user role to the root project
          and the leaf project
        - Issue the URL to get role assignments for the root project but
          not the subtree - this should return just the root assignment
        - Issue the URL to get role assignments for the root project and
          it's subtree - this should return both assignments
        - Check that explicitly setting include_subtree to False is the
          equivalent to not including it at all in the query.

        """
        root_id, leaf_id, non_inherited_role_id, unused_role_id = self._setup_hierarchical_projects_scenario()
        non_inher_entity_root = self.build_role_assignment_entity(project_id=root_id, user_id=self.user['id'], role_id=non_inherited_role_id)
        self.put(non_inher_entity_root['links']['assignment'])
        non_inher_entity_leaf = self.build_role_assignment_entity(project_id=leaf_id, user_id=self.user['id'], role_id=non_inherited_role_id)
        self.put(non_inher_entity_leaf['links']['assignment'])
        collection_url = '/role_assignments?scope.project.id=%(project)s' % {'project': root_id}
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, resource_url=collection_url)
        self.assertThat(r.result['role_assignments'], matchers.HasLength(1))
        self.assertRoleAssignmentInListResponse(r, non_inher_entity_root)
        collection_url = '/role_assignments?scope.project.id=%(project)s&include_subtree=True' % {'project': root_id}
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, resource_url=collection_url)
        self.assertThat(r.result['role_assignments'], matchers.HasLength(2))
        self.assertRoleAssignmentInListResponse(r, non_inher_entity_root)
        self.assertRoleAssignmentInListResponse(r, non_inher_entity_leaf)
        collection_url = '/role_assignments?scope.project.id=%(project)s&include_subtree=0' % {'project': root_id}
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, resource_url=collection_url)
        self.assertThat(r.result['role_assignments'], matchers.HasLength(1))
        self.assertRoleAssignmentInListResponse(r, non_inher_entity_root)

    def test_get_effective_role_assignments_for_project_tree(self):
        """Get role_assignment ?project_id=X&include_subtree=True&effective``.

        Test Plan:

        - Create 2 roles and a hierarchy of projects with one root and 4 levels
          of child project
        - Issue the URL to add a non-inherited user role to the root project
          and a level 1 project
        - Issue the URL to add an inherited user role on the level 2 project
        - Issue the URL to get effective role assignments for the level 1
          project and it's subtree - this should return a role (non-inherited)
          on the level 1 project and roles (inherited) on each of the level
          2, 3 and 4 projects

        """
        root_id, leaf_id, non_inherited_role_id, inherited_role_id = self._setup_hierarchical_projects_scenario()
        level2 = unit.new_project_ref(domain_id=self.domain['id'], parent_id=leaf_id)
        level3 = unit.new_project_ref(domain_id=self.domain['id'], parent_id=level2['id'])
        level4 = unit.new_project_ref(domain_id=self.domain['id'], parent_id=level3['id'])
        PROVIDERS.resource_api.create_project(level2['id'], level2)
        PROVIDERS.resource_api.create_project(level3['id'], level3)
        PROVIDERS.resource_api.create_project(level4['id'], level4)
        non_inher_entity_root = self.build_role_assignment_entity(project_id=root_id, user_id=self.user['id'], role_id=non_inherited_role_id)
        self.put(non_inher_entity_root['links']['assignment'])
        non_inher_entity_leaf = self.build_role_assignment_entity(project_id=leaf_id, user_id=self.user['id'], role_id=non_inherited_role_id)
        self.put(non_inher_entity_leaf['links']['assignment'])
        inher_entity = self.build_role_assignment_entity(project_id=level2['id'], user_id=self.user['id'], role_id=inherited_role_id, inherited_to_projects=True)
        self.put(inher_entity['links']['assignment'])
        collection_url = '/role_assignments?scope.project.id=%(project)s&include_subtree=True&effective' % {'project': leaf_id}
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, resource_url=collection_url)
        self.assertThat(r.result['role_assignments'], matchers.HasLength(3))
        self.assertRoleAssignmentNotInListResponse(r, non_inher_entity_root)
        self.assertRoleAssignmentInListResponse(r, non_inher_entity_leaf)
        inher_entity['scope']['project']['id'] = level3['id']
        self.assertRoleAssignmentInListResponse(r, inher_entity)
        inher_entity['scope']['project']['id'] = level4['id']
        self.assertRoleAssignmentInListResponse(r, inher_entity)

    def test_get_inherited_role_assignments_for_project_hierarchy(self):
        """Call ``GET /role_assignments?scope.OS-INHERIT:inherited_to``.

        Test Plan:

        - Create 2 roles
        - Create a hierarchy of projects with one root and one leaf project
        - Issue the URL to add a non-inherited user role to the root project
        - Issue the URL to add an inherited user role to the root project
        - Issue the URL to filter inherited to projects role assignments - this
          should return 1 role (inherited) on the root project.

        """
        root_id, leaf_id, non_inherited_role_id, inherited_role_id = self._setup_hierarchical_projects_scenario()
        non_inher_up_entity = self.build_role_assignment_entity(project_id=root_id, user_id=self.user['id'], role_id=non_inherited_role_id)
        self.put(non_inher_up_entity['links']['assignment'])
        inher_up_entity = self.build_role_assignment_entity(project_id=root_id, user_id=self.user['id'], role_id=inherited_role_id, inherited_to_projects=True)
        self.put(inher_up_entity['links']['assignment'])
        collection_url = '/role_assignments?scope.OS-INHERIT:inherited_to=projects'
        r = self.get(collection_url)
        self.assertValidRoleAssignmentListResponse(r, resource_url=collection_url)
        self.assertRoleAssignmentNotInListResponse(r, non_inher_up_entity)
        self.assertRoleAssignmentInListResponse(r, inher_up_entity)
        non_inher_up_entity = self.build_role_assignment_entity(project_id=leaf_id, user_id=self.user['id'], role_id=non_inherited_role_id)
        self.assertRoleAssignmentNotInListResponse(r, non_inher_up_entity)
        inher_up_entity['scope']['project']['id'] = leaf_id
        self.assertRoleAssignmentNotInListResponse(r, inher_up_entity)