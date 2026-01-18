from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
class InheritanceTests(AssignmentTestHelperMixin):

    def test_role_assignments_user_domain_to_project_inheritance(self):
        test_plan = {'entities': {'domains': {'users': 2, 'projects': 1}, 'roles': 3}, 'assignments': [{'user': 0, 'role': 0, 'domain': 0}, {'user': 0, 'role': 1, 'project': 0}, {'user': 0, 'role': 2, 'domain': 0, 'inherited_to_projects': True}, {'user': 1, 'role': 1, 'project': 0}], 'tests': [{'params': {'user': 0}, 'results': [{'user': 0, 'role': 0, 'domain': 0}, {'user': 0, 'role': 1, 'project': 0}, {'user': 0, 'role': 2, 'domain': 0, 'inherited_to_projects': 'projects'}]}, {'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'domain': 0}, {'user': 0, 'role': 1, 'project': 0}, {'user': 0, 'role': 2, 'project': 0, 'indirect': {'domain': 0}}]}, {'params': {'user': 0, 'project': 0, 'effective': True}, 'results': [{'user': 0, 'role': 1, 'project': 0}, {'user': 0, 'role': 2, 'project': 0, 'indirect': {'domain': 0}}]}]}
        self.execute_assignment_plan(test_plan)

    def _test_crud_inherited_and_direct_assignment(self, **kwargs):
        """Test inherited and direct assignments for the actor and target.

        Ensure it is possible to create both inherited and direct role
        assignments for the same actor on the same target. The actor and the
        target are specified in the kwargs as ('user_id' or 'group_id') and
        ('project_id' or 'domain_id'), respectively.

        """
        role = unit.new_role_ref()
        role = PROVIDERS.role_api.create_role(role['id'], role)
        assignment_entity = {'role_id': role['id']}
        assignment_entity.update(kwargs)
        direct_assignment_entity = assignment_entity.copy()
        inherited_assignment_entity = assignment_entity.copy()
        inherited_assignment_entity['inherited_to_projects'] = 'projects'
        PROVIDERS.assignment_api.create_grant(inherited_to_projects=False, **assignment_entity)
        grants = PROVIDERS.assignment_api.list_role_assignments(role_id=role['id'])
        self.assertThat(grants, matchers.HasLength(1))
        self.assertIn(direct_assignment_entity, grants)
        PROVIDERS.assignment_api.create_grant(inherited_to_projects=True, **assignment_entity)
        grants = PROVIDERS.assignment_api.list_role_assignments(role_id=role['id'])
        self.assertThat(grants, matchers.HasLength(2))
        self.assertIn(direct_assignment_entity, grants)
        self.assertIn(inherited_assignment_entity, grants)
        PROVIDERS.assignment_api.delete_grant(inherited_to_projects=False, **assignment_entity)
        PROVIDERS.assignment_api.delete_grant(inherited_to_projects=True, **assignment_entity)
        grants = PROVIDERS.assignment_api.list_role_assignments(role_id=role['id'])
        self.assertEqual([], grants)

    def test_crud_inherited_and_direct_assignment_for_user_on_domain(self):
        self._test_crud_inherited_and_direct_assignment(user_id=self.user_foo['id'], domain_id=CONF.identity.default_domain_id)

    def test_crud_inherited_and_direct_assignment_for_group_on_domain(self):
        group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group = PROVIDERS.identity_api.create_group(group)
        self._test_crud_inherited_and_direct_assignment(group_id=group['id'], domain_id=CONF.identity.default_domain_id)

    def test_crud_inherited_and_direct_assignment_for_user_on_project(self):
        self._test_crud_inherited_and_direct_assignment(user_id=self.user_foo['id'], project_id=self.project_baz['id'])

    def test_crud_inherited_and_direct_assignment_for_group_on_project(self):
        group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group = PROVIDERS.identity_api.create_group(group)
        self._test_crud_inherited_and_direct_assignment(group_id=group['id'], project_id=self.project_baz['id'])

    def test_inherited_role_grants_for_user(self):
        """Test inherited user roles.

        Test Plan:

        - Enable OS-INHERIT extension
        - Create 3 roles
        - Create a domain, with a project and a user
        - Check no roles yet exit
        - Assign a direct user role to the project and a (non-inherited)
          user role to the domain
        - Get a list of effective roles - should only get the one direct role
        - Now add an inherited user role to the domain
        - Get a list of effective roles - should have two roles, one
          direct and one by virtue of the inherited user role
        - Also get effective roles for the domain - the role marked as
          inherited should not show up

        """
        role_list = []
        for _ in range(3):
            role = unit.new_role_ref()
            PROVIDERS.role_api.create_role(role['id'], role)
            role_list.append(role)
        domain1 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
        user1 = unit.new_user_ref(domain_id=domain1['id'])
        user1 = PROVIDERS.identity_api.create_user(user1)
        project1 = unit.new_project_ref(domain_id=domain1['id'])
        PROVIDERS.resource_api.create_project(project1['id'], project1)
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], project_id=project1['id'])
        self.assertEqual(0, len(roles_ref))
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=project1['id'], role_id=role_list[0]['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], domain_id=domain1['id'], role_id=role_list[1]['id'])
        combined_list = PROVIDERS.assignment_api.get_roles_for_user_and_project(user1['id'], project1['id'])
        self.assertEqual(1, len(combined_list))
        self.assertIn(role_list[0]['id'], combined_list)
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], domain_id=domain1['id'], role_id=role_list[2]['id'], inherited_to_projects=True)
        combined_list = PROVIDERS.assignment_api.get_roles_for_user_and_project(user1['id'], project1['id'])
        self.assertEqual(2, len(combined_list))
        self.assertIn(role_list[0]['id'], combined_list)
        self.assertIn(role_list[2]['id'], combined_list)
        combined_role_list = PROVIDERS.assignment_api.get_roles_for_user_and_domain(user1['id'], domain1['id'])
        self.assertEqual(1, len(combined_role_list))
        self.assertIn(role_list[1]['id'], combined_role_list)
        test_plan = {'entities': {'domains': {'users': 1, 'projects': 1}, 'roles': 3}, 'assignments': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 1, 'domain': 0}, {'user': 0, 'role': 2, 'domain': 0, 'inherited_to_projects': True}], 'tests': [{'params': {'user': 0, 'project': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 2, 'project': 0, 'indirect': {'domain': 0}}]}, {'params': {'user': 0, 'domain': 0, 'effective': True}, 'results': [{'user': 0, 'role': 1, 'domain': 0}]}, {'params': {'user': 0, 'domain': 0, 'inherited': False}, 'results': [{'user': 0, 'role': 1, 'domain': 0}]}]}
        self.execute_assignment_plan(test_plan)

    def test_inherited_role_grants_for_group(self):
        """Test inherited group roles.

        Test Plan:

        - Enable OS-INHERIT extension
        - Create 4 roles
        - Create a domain, with a project, user and two groups
        - Make the user a member of both groups
        - Check no roles yet exit
        - Assign a direct user role to the project and a (non-inherited)
          group role on the domain
        - Get a list of effective roles - should only get the one direct role
        - Now add two inherited group roles to the domain
        - Get a list of effective roles - should have three roles, one
          direct and two by virtue of inherited group roles

        """
        role_list = []
        for _ in range(4):
            role = unit.new_role_ref()
            PROVIDERS.role_api.create_role(role['id'], role)
            role_list.append(role)
        domain1 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
        user1 = unit.new_user_ref(domain_id=domain1['id'])
        user1 = PROVIDERS.identity_api.create_user(user1)
        group1 = unit.new_group_ref(domain_id=domain1['id'])
        group1 = PROVIDERS.identity_api.create_group(group1)
        group2 = unit.new_group_ref(domain_id=domain1['id'])
        group2 = PROVIDERS.identity_api.create_group(group2)
        project1 = unit.new_project_ref(domain_id=domain1['id'])
        PROVIDERS.resource_api.create_project(project1['id'], project1)
        PROVIDERS.identity_api.add_user_to_group(user1['id'], group1['id'])
        PROVIDERS.identity_api.add_user_to_group(user1['id'], group2['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], project_id=project1['id'])
        self.assertEqual(0, len(roles_ref))
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=project1['id'], role_id=role_list[0]['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], domain_id=domain1['id'], role_id=role_list[1]['id'])
        combined_list = PROVIDERS.assignment_api.get_roles_for_user_and_project(user1['id'], project1['id'])
        self.assertEqual(1, len(combined_list))
        self.assertIn(role_list[0]['id'], combined_list)
        PROVIDERS.assignment_api.create_grant(group_id=group2['id'], domain_id=domain1['id'], role_id=role_list[2]['id'], inherited_to_projects=True)
        PROVIDERS.assignment_api.create_grant(group_id=group2['id'], domain_id=domain1['id'], role_id=role_list[3]['id'], inherited_to_projects=True)
        combined_list = PROVIDERS.assignment_api.get_roles_for_user_and_project(user1['id'], project1['id'])
        self.assertEqual(3, len(combined_list))
        self.assertIn(role_list[0]['id'], combined_list)
        self.assertIn(role_list[2]['id'], combined_list)
        self.assertIn(role_list[3]['id'], combined_list)
        test_plan = {'entities': {'domains': {'users': 1, 'projects': 1, 'groups': 2}, 'roles': 4}, 'group_memberships': [{'group': 0, 'users': [0]}, {'group': 1, 'users': [0]}], 'assignments': [{'user': 0, 'role': 0, 'project': 0}, {'group': 0, 'role': 1, 'domain': 0}, {'group': 1, 'role': 2, 'domain': 0, 'inherited_to_projects': True}, {'group': 1, 'role': 3, 'domain': 0, 'inherited_to_projects': True}], 'tests': [{'params': {'user': 0, 'project': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 2, 'project': 0, 'indirect': {'domain': 0, 'group': 1}}, {'user': 0, 'role': 3, 'project': 0, 'indirect': {'domain': 0, 'group': 1}}]}]}
        self.execute_assignment_plan(test_plan)

    def test_list_projects_for_user_with_inherited_grants(self):
        """Test inherited user roles.

        Test Plan:

        - Enable OS-INHERIT extension
        - Create a domain, with two projects and a user
        - Assign an inherited user role on the domain, as well as a direct
          user role to a separate project in a different domain
        - Get a list of projects for user, should return all three projects

        """
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        user1 = unit.new_user_ref(domain_id=domain['id'])
        user1 = PROVIDERS.identity_api.create_user(user1)
        project1 = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project1['id'], project1)
        project2 = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project2['id'], project2)
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=self.project_bar['id'], role_id=self.role_member['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], domain_id=domain['id'], role_id=self.role_admin['id'], inherited_to_projects=True)
        user_projects = PROVIDERS.assignment_api.list_projects_for_user(user1['id'])
        self.assertEqual(3, len(user_projects))
        test_plan = {'entities': {'domains': [{'projects': 1}, {'users': 1, 'projects': 2}], 'roles': 2}, 'assignments': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 1, 'domain': 1, 'inherited_to_projects': True}], 'tests': [{'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 1, 'project': 1, 'indirect': {'domain': 1}}, {'user': 0, 'role': 1, 'project': 2, 'indirect': {'domain': 1}}]}]}
        self.execute_assignment_plan(test_plan)

    def test_list_projects_for_user_with_inherited_user_project_grants(self):
        """Test inherited role assignments for users on nested projects.

        Test Plan:

        - Enable OS-INHERIT extension
        - Create a hierarchy of projects with one root and one leaf project
        - Assign an inherited user role on root project
        - Assign a non-inherited user role on root project
        - Get a list of projects for user, should return both projects
        - Disable OS-INHERIT extension
        - Get a list of projects for user, should return only root project

        """
        root_project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        root_project = PROVIDERS.resource_api.create_project(root_project['id'], root_project)
        leaf_project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id, parent_id=root_project['id'])
        leaf_project = PROVIDERS.resource_api.create_project(leaf_project['id'], leaf_project)
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        PROVIDERS.assignment_api.create_grant(user_id=user['id'], project_id=root_project['id'], role_id=self.role_admin['id'], inherited_to_projects=True)
        PROVIDERS.assignment_api.create_grant(user_id=user['id'], project_id=root_project['id'], role_id=self.role_member['id'])
        user_projects = PROVIDERS.assignment_api.list_projects_for_user(user['id'])
        self.assertEqual(2, len(user_projects))
        self.assertIn(root_project, user_projects)
        self.assertIn(leaf_project, user_projects)
        test_plan = {'entities': {'domains': {'id': CONF.identity.default_domain_id, 'users': 1, 'projects': {'project': 1}}, 'roles': 2}, 'assignments': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 1, 'project': 0, 'inherited_to_projects': True}], 'tests': [{'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 1, 'project': 1, 'indirect': {'project': 0}}]}]}
        self.execute_assignment_plan(test_plan)

    def test_list_projects_for_user_with_inherited_group_grants(self):
        """Test inherited group roles.

        Test Plan:

        - Enable OS-INHERIT extension
        - Create two domains, each with two projects
        - Create a user and group
        - Make the user a member of the group
        - Assign a user role two projects, an inherited
          group role to one domain and an inherited regular role on
          the other domain
        - Get a list of projects for user, should return both pairs of projects
          from the domain, plus the one separate project

        """
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        domain2 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
        project1 = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project1['id'], project1)
        project2 = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project2['id'], project2)
        project3 = unit.new_project_ref(domain_id=domain2['id'])
        PROVIDERS.resource_api.create_project(project3['id'], project3)
        project4 = unit.new_project_ref(domain_id=domain2['id'])
        PROVIDERS.resource_api.create_project(project4['id'], project4)
        user1 = unit.new_user_ref(domain_id=domain['id'])
        user1 = PROVIDERS.identity_api.create_user(user1)
        group1 = unit.new_group_ref(domain_id=domain['id'])
        group1 = PROVIDERS.identity_api.create_group(group1)
        PROVIDERS.identity_api.add_user_to_group(user1['id'], group1['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=project3['id'], role_id=self.role_member['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=self.project_bar['id'], role_id=self.role_member['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], domain_id=domain['id'], role_id=self.role_admin['id'], inherited_to_projects=True)
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], domain_id=domain2['id'], role_id=self.role_admin['id'], inherited_to_projects=True)
        user_projects = PROVIDERS.assignment_api.list_projects_for_user(user1['id'])
        self.assertEqual(5, len(user_projects))
        test_plan = {'entities': {'domains': [{'projects': 1}, {'users': 1, 'groups': 1, 'projects': 2}, {'projects': 2}], 'roles': 2}, 'group_memberships': [{'group': 0, 'users': [0]}], 'assignments': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 0, 'project': 3}, {'user': 0, 'role': 1, 'domain': 1, 'inherited_to_projects': True}, {'user': 0, 'role': 1, 'domain': 2, 'inherited_to_projects': True}], 'tests': [{'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'project': 3}, {'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 1, 'project': 1, 'indirect': {'domain': 1}}, {'user': 0, 'role': 1, 'project': 2, 'indirect': {'domain': 1}}, {'user': 0, 'role': 1, 'project': 3, 'indirect': {'domain': 2}}, {'user': 0, 'role': 1, 'project': 4, 'indirect': {'domain': 2}}]}]}
        self.execute_assignment_plan(test_plan)

    def test_list_projects_for_user_with_inherited_group_project_grants(self):
        """Test inherited role assignments for groups on nested projects.

        Test Plan:

        - Enable OS-INHERIT extension
        - Create a hierarchy of projects with one root and one leaf project
        - Assign an inherited group role on root project
        - Assign a non-inherited group role on root project
        - Get a list of projects for user, should return both projects
        - Disable OS-INHERIT extension
        - Get a list of projects for user, should return only root project

        """
        root_project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        root_project = PROVIDERS.resource_api.create_project(root_project['id'], root_project)
        leaf_project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id, parent_id=root_project['id'])
        leaf_project = PROVIDERS.resource_api.create_project(leaf_project['id'], leaf_project)
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group = PROVIDERS.identity_api.create_group(group)
        PROVIDERS.identity_api.add_user_to_group(user['id'], group['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group['id'], project_id=root_project['id'], role_id=self.role_admin['id'], inherited_to_projects=True)
        PROVIDERS.assignment_api.create_grant(group_id=group['id'], project_id=root_project['id'], role_id=self.role_member['id'])
        user_projects = PROVIDERS.assignment_api.list_projects_for_user(user['id'])
        self.assertEqual(2, len(user_projects))
        self.assertIn(root_project, user_projects)
        self.assertIn(leaf_project, user_projects)
        test_plan = {'entities': {'domains': {'id': CONF.identity.default_domain_id, 'users': 1, 'groups': 1, 'projects': {'project': 1}}, 'roles': 2}, 'group_memberships': [{'group': 0, 'users': [0]}], 'assignments': [{'group': 0, 'role': 0, 'project': 0}, {'group': 0, 'role': 1, 'project': 0, 'inherited_to_projects': True}], 'tests': [{'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'project': 0, 'indirect': {'group': 0}}, {'user': 0, 'role': 1, 'project': 1, 'indirect': {'group': 0, 'project': 0}}]}]}
        self.execute_assignment_plan(test_plan)

    def test_list_assignments_for_tree(self):
        """Test we correctly list direct assignments for a tree."""
        test_plan = {'entities': {'domains': {'projects': {'project': [{'project': 2}, {'project': 2}]}, 'users': 1}, 'roles': 4}, 'assignments': [{'user': 0, 'role': 0, 'project': 1}, {'user': 0, 'role': 1, 'project': 2}, {'user': 0, 'role': 2, 'project': 1, 'inherited_to_projects': True}, {'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 3, 'project': 4}], 'tests': [{'params': {'project': 1, 'include_subtree': True}, 'results': [{'user': 0, 'role': 0, 'project': 1}, {'user': 0, 'role': 1, 'project': 2}, {'user': 0, 'role': 2, 'project': 1, 'inherited_to_projects': 'projects'}]}]}
        self.execute_assignment_plan(test_plan)

    def test_list_effective_assignments_for_tree(self):
        """Test we correctly list effective assignments for a tree."""
        test_plan = {'entities': {'domains': {'projects': {'project': [{'project': 2}, {'project': 2}]}, 'users': 1}, 'roles': 4}, 'assignments': [{'user': 0, 'role': 1, 'project': 1, 'inherited_to_projects': True}, {'user': 0, 'role': 2, 'project': 2}, {'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 3, 'project': 4}], 'tests': [{'params': {'project': 1, 'effective': True, 'include_subtree': True}, 'results': [{'user': 0, 'role': 1, 'project': 2, 'indirect': {'project': 1}}, {'user': 0, 'role': 1, 'project': 3, 'indirect': {'project': 1}}, {'user': 0, 'role': 2, 'project': 2}]}]}
        self.execute_assignment_plan(test_plan)

    def test_list_effective_assignments_for_tree_with_mixed_assignments(self):
        """Test that we correctly combine assignments for a tree.

        In this test we want to ensure that when asking for a list of
        assignments in a subtree, any assignments inherited from above the
        subtree are correctly combined with any assignments within the subtree
        itself.

        """
        test_plan = {'entities': {'domains': {'projects': {'project': [{'project': 2}, {'project': 2}]}, 'users': 2, 'groups': 1}, 'roles': 4}, 'group_memberships': [{'group': 0, 'users': [0, 1]}], 'assignments': [{'user': 0, 'role': 0, 'project': 0, 'inherited_to_projects': True}, {'group': 0, 'role': 1, 'project': 0, 'inherited_to_projects': True}, {'user': 0, 'role': 2, 'project': 1, 'inherited_to_projects': True}, {'user': 0, 'role': 3, 'project': 2}, {'user': 0, 'role': 2, 'project': 5}, {'user': 0, 'role': 3, 'project': 4}], 'tests': [{'params': {'project': 1, 'user': 0, 'effective': True, 'include_subtree': True}, 'results': [{'user': 0, 'role': 0, 'project': 1, 'indirect': {'project': 0}}, {'user': 0, 'role': 0, 'project': 2, 'indirect': {'project': 0}}, {'user': 0, 'role': 0, 'project': 3, 'indirect': {'project': 0}}, {'user': 0, 'role': 1, 'project': 1, 'indirect': {'project': 0, 'group': 0}}, {'user': 0, 'role': 1, 'project': 2, 'indirect': {'project': 0, 'group': 0}}, {'user': 0, 'role': 1, 'project': 3, 'indirect': {'project': 0, 'group': 0}}, {'user': 0, 'role': 2, 'project': 2, 'indirect': {'project': 1}}, {'user': 0, 'role': 2, 'project': 3, 'indirect': {'project': 1}}, {'user': 0, 'role': 3, 'project': 2}]}]}
        self.execute_assignment_plan(test_plan)

    def test_list_effective_assignments_for_tree_with_domain_assignments(self):
        """Test we correctly honor domain inherited assignments on the tree."""
        test_plan = {'entities': {'domains': {'projects': {'project': [{'project': 2}, {'project': 2}]}, 'users': 1}, 'roles': 4}, 'assignments': [{'user': 0, 'role': 1, 'domain': 0, 'inherited_to_projects': True}, {'user': 0, 'role': 2, 'project': 2}, {'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 3, 'project': 4}], 'tests': [{'params': {'project': 1, 'effective': True, 'include_subtree': True}, 'results': [{'user': 0, 'role': 1, 'project': 1, 'indirect': {'domain': 0}}, {'user': 0, 'role': 1, 'project': 2, 'indirect': {'domain': 0}}, {'user': 0, 'role': 1, 'project': 3, 'indirect': {'domain': 0}}, {'user': 0, 'role': 2, 'project': 2}]}]}
        self.execute_assignment_plan(test_plan)

    def test_list_user_ids_for_project_with_inheritance(self):
        test_plan = {'entities': {'domains': {'id': CONF.identity.default_domain_id, 'users': 4, 'groups': 2, 'projects': {'project': 1}}, 'roles': 4}, 'group_memberships': [{'group': 0, 'users': [1]}, {'group': 1, 'users': [3]}], 'assignments': [{'user': 0, 'role': 0, 'project': 1}, {'group': 0, 'role': 1, 'project': 1}, {'user': 2, 'role': 2, 'project': 0, 'inherited_to_projects': True}, {'group': 1, 'role': 3, 'project': 0, 'inherited_to_projects': True}]}
        test_data = self.execute_assignment_plan(test_plan)
        user_ids = PROVIDERS.assignment_api.list_user_ids_for_project(test_data['projects'][1]['id'])
        self.assertThat(user_ids, matchers.HasLength(4))
        for x in range(0, 4):
            self.assertIn(test_data['users'][x]['id'], user_ids)

    def test_list_role_assignment_using_inherited_sourced_groups(self):
        """Test listing inherited assignments when restricted by groups."""
        test_plan = {'entities': {'domains': [{'users': 3, 'groups': 3, 'projects': 3}, 1], 'roles': 3}, 'group_memberships': [{'group': 0, 'users': [0, 1]}, {'group': 1, 'users': [0]}], 'assignments': [{'user': 0, 'role': 0, 'domain': 0}, {'group': 0, 'role': 1, 'domain': 1}, {'group': 1, 'role': 2, 'domain': 0, 'inherited_to_projects': True}, {'group': 1, 'role': 2, 'project': 1}, {'user': 2, 'role': 1, 'project': 1, 'inherited_to_projects': True}, {'group': 2, 'role': 2, 'project': 2}], 'tests': [{'params': {'source_from_group_ids': [0, 1], 'effective': True}, 'results': [{'group': 0, 'role': 1, 'domain': 1}, {'group': 1, 'role': 2, 'project': 0, 'indirect': {'domain': 0}}, {'group': 1, 'role': 2, 'project': 1, 'indirect': {'domain': 0}}, {'group': 1, 'role': 2, 'project': 2, 'indirect': {'domain': 0}}, {'group': 1, 'role': 2, 'project': 1}]}]}
        self.execute_assignment_plan(test_plan)