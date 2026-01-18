from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
class AssignmentTests(AssignmentTestHelperMixin):

    def _get_domain_fixture(self):
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        return domain

    def test_project_add_and_remove_user_role(self):
        user_ids = PROVIDERS.assignment_api.list_user_ids_for_project(self.project_bar['id'])
        self.assertNotIn(self.user_two['id'], user_ids)
        PROVIDERS.assignment_api.add_role_to_user_and_project(project_id=self.project_bar['id'], user_id=self.user_two['id'], role_id=self.role_other['id'])
        user_ids = PROVIDERS.assignment_api.list_user_ids_for_project(self.project_bar['id'])
        self.assertIn(self.user_two['id'], user_ids)
        PROVIDERS.assignment_api.remove_role_from_user_and_project(project_id=self.project_bar['id'], user_id=self.user_two['id'], role_id=self.role_other['id'])
        user_ids = PROVIDERS.assignment_api.list_user_ids_for_project(self.project_bar['id'])
        self.assertNotIn(self.user_two['id'], user_ids)

    def test_remove_user_role_not_assigned(self):
        self.assertRaises(exception.RoleNotFound, PROVIDERS.assignment_api.remove_role_from_user_and_project, project_id=self.project_bar['id'], user_id=self.user_two['id'], role_id=self.role_other['id'])

    def test_list_user_ids_for_project(self):
        user_ids = PROVIDERS.assignment_api.list_user_ids_for_project(self.project_baz['id'])
        self.assertEqual(2, len(user_ids))
        self.assertIn(self.user_two['id'], user_ids)
        self.assertIn(self.user_badguy['id'], user_ids)

    def test_list_user_ids_for_project_no_duplicates(self):
        user_ref = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user_ref = PROVIDERS.identity_api.create_user(user_ref)
        project_ref = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
        for i in range(2):
            role_ref = unit.new_role_ref()
            PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
            PROVIDERS.assignment_api.add_role_to_user_and_project(user_id=user_ref['id'], project_id=project_ref['id'], role_id=role_ref['id'])
        user_ids = PROVIDERS.assignment_api.list_user_ids_for_project(project_ref['id'])
        self.assertEqual(1, len(user_ids))

    def test_get_project_user_ids_returns_not_found(self):
        self.assertRaises(exception.ProjectNotFound, PROVIDERS.assignment_api.list_user_ids_for_project, uuid.uuid4().hex)

    def test_list_role_assignments_unfiltered(self):
        """Test unfiltered listing of role assignments."""
        test_plan = {'entities': {'domains': {'users': 1, 'groups': 1, 'projects': 1}, 'roles': 3}, 'assignments': [{'user': 0, 'role': 0, 'domain': 0}, {'user': 0, 'role': 1, 'project': 0}, {'group': 0, 'role': 2, 'domain': 0}, {'group': 0, 'role': 2, 'project': 0}], 'tests': [{'params': {}, 'results': [{'user': 0, 'role': 0, 'domain': 0}, {'user': 0, 'role': 1, 'project': 0}, {'group': 0, 'role': 2, 'domain': 0}, {'group': 0, 'role': 2, 'project': 0}]}]}
        self.execute_assignment_plan(test_plan)

    def test_list_role_assignments_filtered_by_role(self):
        """Test listing of role assignments filtered by role ID."""
        test_plan = {'entities': {'domains': {'id': CONF.identity.default_domain_id, 'users': 1, 'groups': 1, 'projects': 1}, 'roles': 3}, 'assignments': [{'user': 0, 'role': 0, 'domain': 0}, {'user': 0, 'role': 1, 'project': 0}, {'group': 0, 'role': 2, 'domain': 0}, {'group': 0, 'role': 2, 'project': 0}], 'tests': [{'params': {'role': 2}, 'results': [{'group': 0, 'role': 2, 'domain': 0}, {'group': 0, 'role': 2, 'project': 0}]}]}
        self.execute_assignment_plan(test_plan)

    def test_list_group_role_assignment(self):
        test_plan = {'entities': {'domains': {'id': CONF.identity.default_domain_id, 'groups': 1, 'projects': 1}, 'roles': 1}, 'assignments': [{'group': 0, 'role': 0, 'project': 0}], 'tests': [{'params': {}, 'results': [{'group': 0, 'role': 0, 'project': 0}]}]}
        self.execute_assignment_plan(test_plan)

    def test_list_role_assignments_bad_role(self):
        assignment_list = PROVIDERS.assignment_api.list_role_assignments(role_id=uuid.uuid4().hex)
        self.assertEqual([], assignment_list)

    def test_list_role_assignments_user_not_found(self):

        def _user_not_found(value):
            raise exception.UserNotFound(user_id=value)
        with mock.patch.object(PROVIDERS.identity_api, 'get_user', _user_not_found):
            assignment_list = PROVIDERS.assignment_api.list_role_assignments(include_names=True)
        self.assertNotEqual([], assignment_list)
        for assignment in assignment_list:
            if 'user_name' in assignment:
                self.assertEqual('', assignment['user_name'])
                self.assertEqual('', assignment['user_domain_id'])
                self.assertEqual('', assignment['user_domain_name'])

    def test_list_role_assignments_group_not_found(self):

        def _group_not_found(value):
            raise exception.GroupNotFound(group_id=value)
        for a in PROVIDERS.assignment_api.list_role_assignments():
            PROVIDERS.assignment_api.delete_grant(**a)
        domain_id = CONF.identity.default_domain_id
        group = PROVIDERS.identity_api.create_group(unit.new_group_ref(domain_id=domain_id))
        user1 = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=domain_id))
        user2 = PROVIDERS.identity_api.create_user(unit.new_user_ref(domain_id=domain_id))
        PROVIDERS.identity_api.add_user_to_group(user1['id'], group['id'])
        PROVIDERS.identity_api.add_user_to_group(user2['id'], group['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group['id'], domain_id=domain_id, role_id=default_fixtures.MEMBER_ROLE_ID)
        num_assignments = len(PROVIDERS.assignment_api.list_role_assignments())
        self.assertEqual(1, num_assignments)
        with mock.patch.object(PROVIDERS.identity_api, 'get_group', _group_not_found):
            keystone.assignment.COMPUTED_ASSIGNMENTS_REGION.invalidate()
            assignment_list = PROVIDERS.assignment_api.list_role_assignments(include_names=True)
        self.assertEqual(num_assignments, len(assignment_list))
        for assignment in assignment_list:
            includes_group_assignments = False
            if 'group_name' in assignment:
                includes_group_assignments = True
                self.assertEqual('', assignment['group_name'])
                self.assertEqual('', assignment['group_domain_id'])
                self.assertEqual('', assignment['group_domain_name'])
        self.assertTrue(includes_group_assignments)
        num_effective = len(PROVIDERS.assignment_api.list_role_assignments(effective=True))
        self.assertGreater(num_effective, len(assignment_list))
        with mock.patch.object(PROVIDERS.identity_api, 'list_users_in_group', _group_not_found):
            keystone.assignment.COMPUTED_ASSIGNMENTS_REGION.invalidate()
            assignment_list = PROVIDERS.assignment_api.list_role_assignments(effective=True)
        self.assertGreater(num_effective, len(assignment_list))
        PROVIDERS.assignment_api.delete_grant(group_id=group['id'], domain_id=domain_id, role_id=default_fixtures.MEMBER_ROLE_ID)

    def test_add_duplicate_role_grant(self):
        roles_ref = PROVIDERS.assignment_api.get_roles_for_user_and_project(self.user_foo['id'], self.project_bar['id'])
        self.assertNotIn(self.role_admin['id'], roles_ref)
        PROVIDERS.assignment_api.add_role_to_user_and_project(self.user_foo['id'], self.project_bar['id'], self.role_admin['id'])
        self.assertRaises(exception.Conflict, PROVIDERS.assignment_api.add_role_to_user_and_project, self.user_foo['id'], self.project_bar['id'], self.role_admin['id'])

    def test_get_role_by_user_and_project_with_user_in_group(self):
        """Test for get role by user and project, user was added into a group.

        Test Plan:

        - Create a user, a project & a group, add this user to group
        - Create roles and grant them to user and project
        - Check the role list get by the user and project was as expected

        """
        user_ref = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user_ref = PROVIDERS.identity_api.create_user(user_ref)
        project_ref = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        PROVIDERS.resource_api.create_project(project_ref['id'], project_ref)
        group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group_id = PROVIDERS.identity_api.create_group(group)['id']
        PROVIDERS.identity_api.add_user_to_group(user_ref['id'], group_id)
        role_ref_list = []
        for i in range(2):
            role_ref = unit.new_role_ref()
            PROVIDERS.role_api.create_role(role_ref['id'], role_ref)
            role_ref_list.append(role_ref)
            PROVIDERS.assignment_api.add_role_to_user_and_project(user_id=user_ref['id'], project_id=project_ref['id'], role_id=role_ref['id'])
        role_list = PROVIDERS.assignment_api.get_roles_for_user_and_project(user_ref['id'], project_ref['id'])
        self.assertEqual(set([r['id'] for r in role_ref_list]), set(role_list))

    def test_get_role_by_user_and_project(self):
        roles_ref = PROVIDERS.assignment_api.get_roles_for_user_and_project(self.user_foo['id'], self.project_bar['id'])
        self.assertNotIn(self.role_admin['id'], roles_ref)
        PROVIDERS.assignment_api.add_role_to_user_and_project(self.user_foo['id'], self.project_bar['id'], self.role_admin['id'])
        roles_ref = PROVIDERS.assignment_api.get_roles_for_user_and_project(self.user_foo['id'], self.project_bar['id'])
        self.assertIn(self.role_admin['id'], roles_ref)
        self.assertNotIn(default_fixtures.MEMBER_ROLE_ID, roles_ref)
        PROVIDERS.assignment_api.add_role_to_user_and_project(self.user_foo['id'], self.project_bar['id'], default_fixtures.MEMBER_ROLE_ID)
        roles_ref = PROVIDERS.assignment_api.get_roles_for_user_and_project(self.user_foo['id'], self.project_bar['id'])
        self.assertIn(self.role_admin['id'], roles_ref)
        self.assertIn(default_fixtures.MEMBER_ROLE_ID, roles_ref)

    def test_get_role_by_trustor_and_project(self):
        new_domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
        new_user = unit.new_user_ref(domain_id=new_domain['id'])
        new_user = PROVIDERS.identity_api.create_user(new_user)
        new_project = unit.new_project_ref(domain_id=new_domain['id'])
        PROVIDERS.resource_api.create_project(new_project['id'], new_project)
        role = self._create_role(domain_id=new_domain['id'])
        PROVIDERS.assignment_api.create_grant(user_id=new_user['id'], project_id=new_project['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        PROVIDERS.assignment_api.create_grant(user_id=new_user['id'], domain_id=new_domain['id'], role_id=role['id'], inherited_to_projects=True)
        roles_ids = PROVIDERS.assignment_api.get_roles_for_trustor_and_project(new_user['id'], new_project['id'])
        self.assertEqual(2, len(roles_ids))
        self.assertIn(self.role_member['id'], roles_ids)
        self.assertIn(role['id'], roles_ids)

    def test_get_roles_for_user_and_domain(self):
        """Test for getting roles for user on a domain.

        Test Plan:

        - Create a domain, with 2 users
        - Check no roles yet exit
        - Give user1 two roles on the domain, user2 one role
        - Get roles on user1 and the domain - maybe sure we only
          get back the 2 roles on user1
        - Delete both roles from user1
        - Check we get no roles back for user1 on domain

        """
        new_domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
        new_user1 = unit.new_user_ref(domain_id=new_domain['id'])
        new_user1 = PROVIDERS.identity_api.create_user(new_user1)
        new_user2 = unit.new_user_ref(domain_id=new_domain['id'])
        new_user2 = PROVIDERS.identity_api.create_user(new_user2)
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=new_user1['id'], domain_id=new_domain['id'])
        self.assertEqual(0, len(roles_ref))
        PROVIDERS.assignment_api.create_grant(user_id=new_user1['id'], domain_id=new_domain['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        PROVIDERS.assignment_api.create_grant(user_id=new_user1['id'], domain_id=new_domain['id'], role_id=default_fixtures.OTHER_ROLE_ID)
        PROVIDERS.assignment_api.create_grant(user_id=new_user2['id'], domain_id=new_domain['id'], role_id=default_fixtures.ADMIN_ROLE_ID)
        roles_ids = PROVIDERS.assignment_api.get_roles_for_user_and_domain(new_user1['id'], new_domain['id'])
        self.assertEqual(2, len(roles_ids))
        self.assertIn(self.role_member['id'], roles_ids)
        self.assertIn(self.role_other['id'], roles_ids)
        PROVIDERS.assignment_api.delete_grant(user_id=new_user1['id'], domain_id=new_domain['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        PROVIDERS.assignment_api.delete_grant(user_id=new_user1['id'], domain_id=new_domain['id'], role_id=default_fixtures.OTHER_ROLE_ID)
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=new_user1['id'], domain_id=new_domain['id'])
        self.assertEqual(0, len(roles_ref))

    def test_get_roles_for_user_and_domain_returns_not_found(self):
        """Test errors raised when getting roles for user on a domain.

        Test Plan:

        - Check non-existing user gives UserNotFound
        - Check non-existing domain gives DomainNotFound

        """
        new_domain = self._get_domain_fixture()
        new_user1 = unit.new_user_ref(domain_id=new_domain['id'])
        new_user1 = PROVIDERS.identity_api.create_user(new_user1)
        self.assertRaises(exception.UserNotFound, PROVIDERS.assignment_api.get_roles_for_user_and_domain, uuid.uuid4().hex, new_domain['id'])
        self.assertRaises(exception.DomainNotFound, PROVIDERS.assignment_api.get_roles_for_user_and_domain, new_user1['id'], uuid.uuid4().hex)

    def test_get_roles_for_user_and_project_returns_not_found(self):
        self.assertRaises(exception.UserNotFound, PROVIDERS.assignment_api.get_roles_for_user_and_project, uuid.uuid4().hex, self.project_bar['id'])
        self.assertRaises(exception.ProjectNotFound, PROVIDERS.assignment_api.get_roles_for_user_and_project, self.user_foo['id'], uuid.uuid4().hex)

    def test_add_role_to_user_and_project_returns_not_found(self):
        self.assertRaises(exception.ProjectNotFound, PROVIDERS.assignment_api.add_role_to_user_and_project, self.user_foo['id'], uuid.uuid4().hex, self.role_admin['id'])
        self.assertRaises(exception.RoleNotFound, PROVIDERS.assignment_api.add_role_to_user_and_project, self.user_foo['id'], self.project_bar['id'], uuid.uuid4().hex)

    def test_add_role_to_user_and_project_no_user(self):
        user_id_not_exist = uuid.uuid4().hex
        PROVIDERS.assignment_api.add_role_to_user_and_project(user_id_not_exist, self.project_bar['id'], self.role_admin['id'])

    def test_remove_role_from_user_and_project(self):
        PROVIDERS.assignment_api.add_role_to_user_and_project(self.user_foo['id'], self.project_bar['id'], default_fixtures.MEMBER_ROLE_ID)
        PROVIDERS.assignment_api.remove_role_from_user_and_project(self.user_foo['id'], self.project_bar['id'], default_fixtures.MEMBER_ROLE_ID)
        roles_ref = PROVIDERS.assignment_api.get_roles_for_user_and_project(self.user_foo['id'], self.project_bar['id'])
        self.assertNotIn(default_fixtures.MEMBER_ROLE_ID, roles_ref)
        self.assertRaises(exception.NotFound, PROVIDERS.assignment_api.remove_role_from_user_and_project, self.user_foo['id'], self.project_bar['id'], default_fixtures.MEMBER_ROLE_ID)

    def test_get_role_grant_by_user_and_project(self):
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=self.user_foo['id'], project_id=self.project_bar['id'])
        self.assertEqual(1, len(roles_ref))
        PROVIDERS.assignment_api.create_grant(user_id=self.user_foo['id'], project_id=self.project_bar['id'], role_id=self.role_admin['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=self.user_foo['id'], project_id=self.project_bar['id'])
        self.assertIn(self.role_admin['id'], [role_ref['id'] for role_ref in roles_ref])
        PROVIDERS.assignment_api.create_grant(user_id=self.user_foo['id'], project_id=self.project_bar['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=self.user_foo['id'], project_id=self.project_bar['id'])
        roles_ref_ids = []
        for ref in roles_ref:
            roles_ref_ids.append(ref['id'])
        self.assertIn(self.role_admin['id'], roles_ref_ids)
        self.assertIn(default_fixtures.MEMBER_ROLE_ID, roles_ref_ids)

    def test_remove_role_grant_from_user_and_project(self):
        PROVIDERS.assignment_api.create_grant(user_id=self.user_foo['id'], project_id=self.project_baz['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=self.user_foo['id'], project_id=self.project_baz['id'])
        self.assertDictEqual(self.role_member, roles_ref[0])
        PROVIDERS.assignment_api.delete_grant(user_id=self.user_foo['id'], project_id=self.project_baz['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=self.user_foo['id'], project_id=self.project_baz['id'])
        self.assertEqual(0, len(roles_ref))
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_grant, user_id=self.user_foo['id'], project_id=self.project_baz['id'], role_id=default_fixtures.MEMBER_ROLE_ID)

    def test_get_role_assignment_by_project_not_found(self):
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.check_grant_role_id, user_id=self.user_foo['id'], project_id=self.project_baz['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.check_grant_role_id, group_id=uuid.uuid4().hex, project_id=self.project_baz['id'], role_id=default_fixtures.MEMBER_ROLE_ID)

    def test_get_role_assignment_by_domain_not_found(self):
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.check_grant_role_id, user_id=self.user_foo['id'], domain_id=CONF.identity.default_domain_id, role_id=default_fixtures.MEMBER_ROLE_ID)
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.check_grant_role_id, group_id=uuid.uuid4().hex, domain_id=CONF.identity.default_domain_id, role_id=default_fixtures.MEMBER_ROLE_ID)

    def test_del_role_assignment_by_project_not_found(self):
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_grant, user_id=self.user_foo['id'], project_id=self.project_baz['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_grant, group_id=uuid.uuid4().hex, project_id=self.project_baz['id'], role_id=default_fixtures.MEMBER_ROLE_ID)

    def test_del_role_assignment_by_domain_not_found(self):
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_grant, user_id=self.user_foo['id'], domain_id=CONF.identity.default_domain_id, role_id=default_fixtures.MEMBER_ROLE_ID)
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_grant, group_id=uuid.uuid4().hex, domain_id=CONF.identity.default_domain_id, role_id=default_fixtures.MEMBER_ROLE_ID)

    def test_get_and_remove_role_grant_by_group_and_project(self):
        new_domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
        new_group = unit.new_group_ref(domain_id=new_domain['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group)
        new_user = unit.new_user_ref(domain_id=new_domain['id'])
        new_user = PROVIDERS.identity_api.create_user(new_user)
        PROVIDERS.identity_api.add_user_to_group(new_user['id'], new_group['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], project_id=self.project_bar['id'])
        self.assertEqual(0, len(roles_ref))
        PROVIDERS.assignment_api.create_grant(group_id=new_group['id'], project_id=self.project_bar['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], project_id=self.project_bar['id'])
        self.assertDictEqual(self.role_member, roles_ref[0])
        PROVIDERS.assignment_api.delete_grant(group_id=new_group['id'], project_id=self.project_bar['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], project_id=self.project_bar['id'])
        self.assertEqual(0, len(roles_ref))
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_grant, group_id=new_group['id'], project_id=self.project_bar['id'], role_id=default_fixtures.MEMBER_ROLE_ID)

    def test_get_and_remove_role_grant_by_group_and_domain(self):
        new_domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
        new_group = unit.new_group_ref(domain_id=new_domain['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group)
        new_user = unit.new_user_ref(domain_id=new_domain['id'])
        new_user = PROVIDERS.identity_api.create_user(new_user)
        PROVIDERS.identity_api.add_user_to_group(new_user['id'], new_group['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], domain_id=new_domain['id'])
        self.assertEqual(0, len(roles_ref))
        PROVIDERS.assignment_api.create_grant(group_id=new_group['id'], domain_id=new_domain['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], domain_id=new_domain['id'])
        self.assertDictEqual(self.role_member, roles_ref[0])
        PROVIDERS.assignment_api.delete_grant(group_id=new_group['id'], domain_id=new_domain['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], domain_id=new_domain['id'])
        self.assertEqual(0, len(roles_ref))
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_grant, group_id=new_group['id'], domain_id=new_domain['id'], role_id=default_fixtures.MEMBER_ROLE_ID)

    def test_get_and_remove_correct_role_grant_from_a_mix(self):
        new_domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
        new_project = unit.new_project_ref(domain_id=new_domain['id'])
        PROVIDERS.resource_api.create_project(new_project['id'], new_project)
        new_group = unit.new_group_ref(domain_id=new_domain['id'])
        new_group = PROVIDERS.identity_api.create_group(new_group)
        new_group2 = unit.new_group_ref(domain_id=new_domain['id'])
        new_group2 = PROVIDERS.identity_api.create_group(new_group2)
        new_user = unit.new_user_ref(domain_id=new_domain['id'])
        new_user = PROVIDERS.identity_api.create_user(new_user)
        new_user2 = unit.new_user_ref(domain_id=new_domain['id'])
        new_user2 = PROVIDERS.identity_api.create_user(new_user2)
        PROVIDERS.identity_api.add_user_to_group(new_user['id'], new_group['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], domain_id=new_domain['id'])
        self.assertEqual(0, len(roles_ref))
        PROVIDERS.assignment_api.create_grant(group_id=new_group['id'], domain_id=new_domain['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        PROVIDERS.assignment_api.create_grant(group_id=new_group2['id'], domain_id=new_domain['id'], role_id=self.role_admin['id'])
        PROVIDERS.assignment_api.create_grant(user_id=new_user2['id'], domain_id=new_domain['id'], role_id=self.role_admin['id'])
        PROVIDERS.assignment_api.create_grant(group_id=new_group['id'], project_id=new_project['id'], role_id=self.role_admin['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], domain_id=new_domain['id'])
        self.assertDictEqual(self.role_member, roles_ref[0])
        PROVIDERS.assignment_api.delete_grant(group_id=new_group['id'], domain_id=new_domain['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=new_group['id'], domain_id=new_domain['id'])
        self.assertEqual(0, len(roles_ref))
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_grant, group_id=new_group['id'], domain_id=new_domain['id'], role_id=default_fixtures.MEMBER_ROLE_ID)

    def test_get_and_remove_role_grant_by_user_and_domain(self):
        new_domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
        new_user = unit.new_user_ref(domain_id=new_domain['id'])
        new_user = PROVIDERS.identity_api.create_user(new_user)
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=new_user['id'], domain_id=new_domain['id'])
        self.assertEqual(0, len(roles_ref))
        PROVIDERS.assignment_api.create_grant(user_id=new_user['id'], domain_id=new_domain['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=new_user['id'], domain_id=new_domain['id'])
        self.assertDictEqual(self.role_member, roles_ref[0])
        PROVIDERS.assignment_api.delete_grant(user_id=new_user['id'], domain_id=new_domain['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=new_user['id'], domain_id=new_domain['id'])
        self.assertEqual(0, len(roles_ref))
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_grant, user_id=new_user['id'], domain_id=new_domain['id'], role_id=default_fixtures.MEMBER_ROLE_ID)

    def test_get_and_remove_role_grant_by_group_and_cross_domain(self):
        group1_domain1_role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(group1_domain1_role['id'], group1_domain1_role)
        group1_domain2_role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(group1_domain2_role['id'], group1_domain2_role)
        domain1 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
        domain2 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
        group1 = unit.new_group_ref(domain_id=domain1['id'])
        group1 = PROVIDERS.identity_api.create_group(group1)
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], domain_id=domain1['id'])
        self.assertEqual(0, len(roles_ref))
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], domain_id=domain2['id'])
        self.assertEqual(0, len(roles_ref))
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], domain_id=domain1['id'], role_id=group1_domain1_role['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], domain_id=domain2['id'], role_id=group1_domain2_role['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], domain_id=domain1['id'])
        self.assertDictEqual(group1_domain1_role, roles_ref[0])
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], domain_id=domain2['id'])
        self.assertDictEqual(group1_domain2_role, roles_ref[0])
        PROVIDERS.assignment_api.delete_grant(group_id=group1['id'], domain_id=domain2['id'], role_id=group1_domain2_role['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], domain_id=domain2['id'])
        self.assertEqual(0, len(roles_ref))
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_grant, group_id=group1['id'], domain_id=domain2['id'], role_id=group1_domain2_role['id'])

    def test_get_and_remove_role_grant_by_user_and_cross_domain(self):
        user1_domain1_role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(user1_domain1_role['id'], user1_domain1_role)
        user1_domain2_role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(user1_domain2_role['id'], user1_domain2_role)
        domain1 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
        domain2 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
        user1 = unit.new_user_ref(domain_id=domain1['id'])
        user1 = PROVIDERS.identity_api.create_user(user1)
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], domain_id=domain1['id'])
        self.assertEqual(0, len(roles_ref))
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], domain_id=domain2['id'])
        self.assertEqual(0, len(roles_ref))
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], domain_id=domain1['id'], role_id=user1_domain1_role['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], domain_id=domain2['id'], role_id=user1_domain2_role['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], domain_id=domain1['id'])
        self.assertDictEqual(user1_domain1_role, roles_ref[0])
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], domain_id=domain2['id'])
        self.assertDictEqual(user1_domain2_role, roles_ref[0])
        PROVIDERS.assignment_api.delete_grant(user_id=user1['id'], domain_id=domain2['id'], role_id=user1_domain2_role['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], domain_id=domain2['id'])
        self.assertEqual(0, len(roles_ref))
        self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_grant, user_id=user1['id'], domain_id=domain2['id'], role_id=user1_domain2_role['id'])

    def test_role_grant_by_group_and_cross_domain_project(self):
        role1 = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role1['id'], role1)
        role2 = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role2['id'], role2)
        domain1 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
        domain2 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
        group1 = unit.new_group_ref(domain_id=domain1['id'])
        group1 = PROVIDERS.identity_api.create_group(group1)
        project1 = unit.new_project_ref(domain_id=domain2['id'])
        PROVIDERS.resource_api.create_project(project1['id'], project1)
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], project_id=project1['id'])
        self.assertEqual(0, len(roles_ref))
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], project_id=project1['id'], role_id=role1['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], project_id=project1['id'], role_id=role2['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], project_id=project1['id'])
        roles_ref_ids = []
        for ref in roles_ref:
            roles_ref_ids.append(ref['id'])
        self.assertIn(role1['id'], roles_ref_ids)
        self.assertIn(role2['id'], roles_ref_ids)
        PROVIDERS.assignment_api.delete_grant(group_id=group1['id'], project_id=project1['id'], role_id=role1['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], project_id=project1['id'])
        self.assertEqual(1, len(roles_ref))
        self.assertDictEqual(role2, roles_ref[0])

    def test_role_grant_by_user_and_cross_domain_project(self):
        role1 = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role1['id'], role1)
        role2 = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role2['id'], role2)
        domain1 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
        domain2 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
        user1 = unit.new_user_ref(domain_id=domain1['id'])
        user1 = PROVIDERS.identity_api.create_user(user1)
        project1 = unit.new_project_ref(domain_id=domain2['id'])
        PROVIDERS.resource_api.create_project(project1['id'], project1)
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], project_id=project1['id'])
        self.assertEqual(0, len(roles_ref))
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=project1['id'], role_id=role1['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=project1['id'], role_id=role2['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], project_id=project1['id'])
        roles_ref_ids = []
        for ref in roles_ref:
            roles_ref_ids.append(ref['id'])
        self.assertIn(role1['id'], roles_ref_ids)
        self.assertIn(role2['id'], roles_ref_ids)
        PROVIDERS.assignment_api.delete_grant(user_id=user1['id'], project_id=project1['id'], role_id=role1['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], project_id=project1['id'])
        self.assertEqual(1, len(roles_ref))
        self.assertDictEqual(role2, roles_ref[0])

    def test_delete_user_grant_no_user(self):
        role = unit.new_role_ref()
        role_id = role['id']
        PROVIDERS.role_api.create_role(role_id, role)
        user_id = uuid.uuid4().hex
        PROVIDERS.assignment_api.create_grant(role_id, user_id=user_id, project_id=self.project_bar['id'])
        PROVIDERS.assignment_api.delete_grant(role_id, user_id=user_id, project_id=self.project_bar['id'])

    def test_delete_group_grant_no_group(self):
        role = unit.new_role_ref()
        role_id = role['id']
        PROVIDERS.role_api.create_role(role_id, role)
        group_id = uuid.uuid4().hex
        PROVIDERS.assignment_api.create_grant(role_id, group_id=group_id, project_id=self.project_bar['id'])
        PROVIDERS.assignment_api.delete_grant(role_id, group_id=group_id, project_id=self.project_bar['id'])

    def test_grant_crud_throws_exception_if_invalid_role(self):
        """Ensure RoleNotFound thrown if role does not exist."""

        def assert_role_not_found_exception(f, **kwargs):
            self.assertRaises(exception.RoleNotFound, f, role_id=uuid.uuid4().hex, **kwargs)
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user_resp = PROVIDERS.identity_api.create_user(user)
        group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group_resp = PROVIDERS.identity_api.create_group(group)
        project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        project_resp = PROVIDERS.resource_api.create_project(project['id'], project)
        for manager_call in [PROVIDERS.assignment_api.create_grant, PROVIDERS.assignment_api.get_grant]:
            assert_role_not_found_exception(manager_call, user_id=user_resp['id'], project_id=project_resp['id'])
            assert_role_not_found_exception(manager_call, group_id=group_resp['id'], project_id=project_resp['id'])
            assert_role_not_found_exception(manager_call, user_id=user_resp['id'], domain_id=CONF.identity.default_domain_id)
            assert_role_not_found_exception(manager_call, group_id=group_resp['id'], domain_id=CONF.identity.default_domain_id)
        assert_role_not_found_exception(PROVIDERS.assignment_api.delete_grant, user_id=user_resp['id'], project_id=project_resp['id'])
        assert_role_not_found_exception(PROVIDERS.assignment_api.delete_grant, group_id=group_resp['id'], project_id=project_resp['id'])
        assert_role_not_found_exception(PROVIDERS.assignment_api.delete_grant, user_id=user_resp['id'], domain_id=CONF.identity.default_domain_id)
        assert_role_not_found_exception(PROVIDERS.assignment_api.delete_grant, group_id=group_resp['id'], domain_id=CONF.identity.default_domain_id)

    def test_multi_role_grant_by_user_group_on_project_domain(self):
        role_list = []
        for _ in range(10):
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
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], domain_id=domain1['id'], role_id=role_list[0]['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], domain_id=domain1['id'], role_id=role_list[1]['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], domain_id=domain1['id'], role_id=role_list[2]['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], domain_id=domain1['id'], role_id=role_list[3]['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=project1['id'], role_id=role_list[4]['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=project1['id'], role_id=role_list[5]['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], project_id=project1['id'], role_id=role_list[6]['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], project_id=project1['id'], role_id=role_list[7]['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], domain_id=domain1['id'])
        self.assertEqual(2, len(roles_ref))
        self.assertIn(role_list[0], roles_ref)
        self.assertIn(role_list[1], roles_ref)
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], domain_id=domain1['id'])
        self.assertEqual(2, len(roles_ref))
        self.assertIn(role_list[2], roles_ref)
        self.assertIn(role_list[3], roles_ref)
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], project_id=project1['id'])
        self.assertEqual(2, len(roles_ref))
        self.assertIn(role_list[4], roles_ref)
        self.assertIn(role_list[5], roles_ref)
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], project_id=project1['id'])
        self.assertEqual(2, len(roles_ref))
        self.assertIn(role_list[6], roles_ref)
        self.assertIn(role_list[7], roles_ref)
        combined_list = PROVIDERS.assignment_api.get_roles_for_user_and_project(user1['id'], project1['id'])
        self.assertEqual(4, len(combined_list))
        self.assertIn(role_list[4]['id'], combined_list)
        self.assertIn(role_list[5]['id'], combined_list)
        self.assertIn(role_list[6]['id'], combined_list)
        self.assertIn(role_list[7]['id'], combined_list)
        combined_role_list = PROVIDERS.assignment_api.get_roles_for_user_and_domain(user1['id'], domain1['id'])
        self.assertEqual(4, len(combined_role_list))
        self.assertIn(role_list[0]['id'], combined_role_list)
        self.assertIn(role_list[1]['id'], combined_role_list)
        self.assertIn(role_list[2]['id'], combined_role_list)
        self.assertIn(role_list[3]['id'], combined_role_list)

    def test_multi_group_grants_on_project_domain(self):
        """Test multiple group roles for user on project and domain.

        Test Plan:

        - Create 6 roles
        - Create a domain, with a project, user and two groups
        - Make the user a member of both groups
        - Check no roles yet exit
        - Assign a role to each user and both groups on both the
          project and domain
        - Get a list of effective roles for the user on both the
          project and domain, checking we get back the correct three
          roles

        """
        role_list = []
        for _ in range(6):
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
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], domain_id=domain1['id'], role_id=role_list[0]['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], domain_id=domain1['id'], role_id=role_list[1]['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group2['id'], domain_id=domain1['id'], role_id=role_list[2]['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=project1['id'], role_id=role_list[3]['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], project_id=project1['id'], role_id=role_list[4]['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group2['id'], project_id=project1['id'], role_id=role_list[5]['id'])
        combined_list = PROVIDERS.assignment_api.get_roles_for_user_and_project(user1['id'], project1['id'])
        self.assertEqual(3, len(combined_list))
        self.assertIn(role_list[3]['id'], combined_list)
        self.assertIn(role_list[4]['id'], combined_list)
        self.assertIn(role_list[5]['id'], combined_list)
        combined_role_list = PROVIDERS.assignment_api.get_roles_for_user_and_domain(user1['id'], domain1['id'])
        self.assertEqual(3, len(combined_role_list))
        self.assertIn(role_list[0]['id'], combined_role_list)
        self.assertIn(role_list[1]['id'], combined_role_list)
        self.assertIn(role_list[2]['id'], combined_role_list)

    def test_delete_role_with_user_and_group_grants(self):
        role1 = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role1['id'], role1)
        domain1 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
        project1 = unit.new_project_ref(domain_id=domain1['id'])
        PROVIDERS.resource_api.create_project(project1['id'], project1)
        user1 = unit.new_user_ref(domain_id=domain1['id'])
        user1 = PROVIDERS.identity_api.create_user(user1)
        group1 = unit.new_group_ref(domain_id=domain1['id'])
        group1 = PROVIDERS.identity_api.create_group(group1)
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=project1['id'], role_id=role1['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], domain_id=domain1['id'], role_id=role1['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], project_id=project1['id'], role_id=role1['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], domain_id=domain1['id'], role_id=role1['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], project_id=project1['id'])
        self.assertEqual(1, len(roles_ref))
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], project_id=project1['id'])
        self.assertEqual(1, len(roles_ref))
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], domain_id=domain1['id'])
        self.assertEqual(1, len(roles_ref))
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], domain_id=domain1['id'])
        self.assertEqual(1, len(roles_ref))
        PROVIDERS.role_api.delete_role(role1['id'])
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], project_id=project1['id'])
        self.assertEqual(0, len(roles_ref))
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], project_id=project1['id'])
        self.assertEqual(0, len(roles_ref))
        roles_ref = PROVIDERS.assignment_api.list_grants(user_id=user1['id'], domain_id=domain1['id'])
        self.assertEqual(0, len(roles_ref))
        roles_ref = PROVIDERS.assignment_api.list_grants(group_id=group1['id'], domain_id=domain1['id'])
        self.assertEqual(0, len(roles_ref))

    def test_list_role_assignment_by_domain(self):
        """Test listing of role assignment filtered by domain."""
        test_plan = {'entities': {'domains': [{'users': 3, 'groups': 1}, 1], 'roles': 2}, 'group_memberships': [{'group': 0, 'users': [1, 2]}], 'assignments': [{'user': 0, 'role': 0, 'domain': 0}, {'group': 0, 'role': 1, 'domain': 0}], 'tests': [{'params': {'domain': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'domain': 0}, {'user': 1, 'role': 1, 'domain': 0, 'indirect': {'group': 0}}, {'user': 2, 'role': 1, 'domain': 0, 'indirect': {'group': 0}}]}, {'params': {'domain': 1, 'effective': True}, 'results': []}]}
        self.execute_assignment_plan(test_plan)

    def test_list_role_assignment_by_user_with_domain_group_roles(self):
        """Test listing assignments by user, with group roles on a domain."""
        test_plan = {'entities': {'domains': [{'users': 3, 'groups': 3}, 1], 'roles': 3}, 'group_memberships': [{'group': 0, 'users': [0, 1]}, {'group': 1, 'users': [0]}], 'assignments': [{'user': 0, 'role': 0, 'domain': 0}, {'group': 0, 'role': 1, 'domain': 0}, {'group': 1, 'role': 2, 'domain': 0}, {'user': 1, 'role': 1, 'domain': 0}, {'group': 2, 'role': 2, 'domain': 0}], 'tests': [{'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'domain': 0}, {'user': 0, 'role': 1, 'domain': 0, 'indirect': {'group': 0}}, {'user': 0, 'role': 2, 'domain': 0, 'indirect': {'group': 1}}]}, {'params': {'user': 0, 'domain': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'domain': 0}, {'user': 0, 'role': 1, 'domain': 0, 'indirect': {'group': 0}}, {'user': 0, 'role': 2, 'domain': 0, 'indirect': {'group': 1}}]}, {'params': {'user': 0, 'domain': 1, 'effective': True}, 'results': []}, {'params': {'user': 2, 'domain': 0, 'effective': True}, 'results': []}]}
        self.execute_assignment_plan(test_plan)

    def test_list_role_assignment_using_sourced_groups(self):
        """Test listing assignments when restricted by source groups."""
        test_plan = {'entities': {'domains': {'id': CONF.identity.default_domain_id, 'users': 3, 'groups': 3, 'projects': 3}, 'roles': 3}, 'group_memberships': [{'group': 0, 'users': [0, 1]}, {'group': 1, 'users': [0]}], 'assignments': [{'user': 0, 'role': 0, 'project': 0}, {'group': 0, 'role': 1, 'project': 1}, {'group': 1, 'role': 2, 'project': 0}, {'group': 1, 'role': 2, 'project': 1}, {'user': 2, 'role': 1, 'project': 1}, {'group': 2, 'role': 2, 'project': 2}], 'tests': [{'params': {'source_from_group_ids': [0, 1], 'effective': True}, 'results': [{'group': 0, 'role': 1, 'project': 1}, {'group': 1, 'role': 2, 'project': 0}, {'group': 1, 'role': 2, 'project': 1}]}, {'params': {'source_from_group_ids': [0, 1], 'role': 2, 'effective': True}, 'results': [{'group': 1, 'role': 2, 'project': 0}, {'group': 1, 'role': 2, 'project': 1}]}]}
        self.execute_assignment_plan(test_plan)

    def test_list_role_assignment_using_sourced_groups_with_domains(self):
        """Test listing domain assignments when restricted by source groups."""
        test_plan = {'entities': {'domains': [{'users': 3, 'groups': 3, 'projects': 3}, 1], 'roles': 3}, 'group_memberships': [{'group': 0, 'users': [0, 1]}, {'group': 1, 'users': [0]}], 'assignments': [{'user': 0, 'role': 0, 'domain': 0}, {'group': 0, 'role': 1, 'domain': 1}, {'group': 1, 'role': 2, 'project': 0}, {'group': 1, 'role': 2, 'project': 1}, {'user': 2, 'role': 1, 'project': 1}, {'group': 2, 'role': 2, 'project': 2}], 'tests': [{'params': {'source_from_group_ids': [0, 1], 'effective': True}, 'results': [{'group': 0, 'role': 1, 'domain': 1}, {'group': 1, 'role': 2, 'project': 0}, {'group': 1, 'role': 2, 'project': 1}]}, {'params': {'source_from_group_ids': [0, 1], 'role': 1, 'effective': True}, 'results': [{'group': 0, 'role': 1, 'domain': 1}]}]}
        self.execute_assignment_plan(test_plan)

    def test_list_role_assignment_fails_with_userid_and_source_groups(self):
        """Show we trap this unsupported internal combination of params."""
        group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group = PROVIDERS.identity_api.create_group(group)
        self.assertRaises(exception.UnexpectedError, PROVIDERS.assignment_api.list_role_assignments, effective=True, user_id=self.user_foo['id'], source_from_group_ids=[group['id']])

    def test_list_user_project_ids_returns_not_found(self):
        self.assertRaises(exception.UserNotFound, PROVIDERS.assignment_api.list_projects_for_user, uuid.uuid4().hex)

    def test_delete_user_with_project_association(self):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        role_member = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role_member['id'], role_member)
        PROVIDERS.assignment_api.add_role_to_user_and_project(user['id'], self.project_bar['id'], role_member['id'])
        PROVIDERS.identity_api.delete_user(user['id'])
        self.assertRaises(exception.UserNotFound, PROVIDERS.assignment_api.list_projects_for_user, user['id'])

    def test_delete_user_with_project_roles(self):
        user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.create_user(user)
        PROVIDERS.assignment_api.add_role_to_user_and_project(user['id'], self.project_bar['id'], self.role_member['id'])
        PROVIDERS.identity_api.delete_user(user['id'])
        self.assertRaises(exception.UserNotFound, PROVIDERS.assignment_api.list_projects_for_user, user['id'])

    def test_delete_role_returns_not_found(self):
        self.assertRaises(exception.RoleNotFound, PROVIDERS.role_api.delete_role, uuid.uuid4().hex)

    def test_delete_project_with_role_assignments(self):
        project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        PROVIDERS.resource_api.create_project(project['id'], project)
        PROVIDERS.assignment_api.add_role_to_user_and_project(self.user_foo['id'], project['id'], default_fixtures.MEMBER_ROLE_ID)
        PROVIDERS.resource_api.delete_project(project['id'])
        self.assertRaises(exception.ProjectNotFound, PROVIDERS.assignment_api.list_user_ids_for_project, project['id'])

    def test_delete_role_check_role_grant(self):
        role = unit.new_role_ref()
        alt_role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        PROVIDERS.role_api.create_role(alt_role['id'], alt_role)
        PROVIDERS.assignment_api.add_role_to_user_and_project(self.user_foo['id'], self.project_bar['id'], role['id'])
        PROVIDERS.assignment_api.add_role_to_user_and_project(self.user_foo['id'], self.project_bar['id'], alt_role['id'])
        PROVIDERS.role_api.delete_role(role['id'])
        roles_ref = PROVIDERS.assignment_api.get_roles_for_user_and_project(self.user_foo['id'], self.project_bar['id'])
        self.assertNotIn(role['id'], roles_ref)
        self.assertIn(alt_role['id'], roles_ref)

    def test_list_projects_for_user(self):
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        user1 = unit.new_user_ref(domain_id=domain['id'])
        user1 = PROVIDERS.identity_api.create_user(user1)
        user_projects = PROVIDERS.assignment_api.list_projects_for_user(user1['id'])
        self.assertEqual(0, len(user_projects))
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=self.project_bar['id'], role_id=self.role_member['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=self.project_baz['id'], role_id=self.role_member['id'])
        user_projects = PROVIDERS.assignment_api.list_projects_for_user(user1['id'])
        self.assertEqual(2, len(user_projects))

    def test_list_projects_for_user_with_grants(self):
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        user1 = unit.new_user_ref(domain_id=domain['id'])
        user1 = PROVIDERS.identity_api.create_user(user1)
        group1 = unit.new_group_ref(domain_id=domain['id'])
        group1 = PROVIDERS.identity_api.create_group(group1)
        group2 = unit.new_group_ref(domain_id=domain['id'])
        group2 = PROVIDERS.identity_api.create_group(group2)
        project1 = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project1['id'], project1)
        project2 = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project2['id'], project2)
        PROVIDERS.identity_api.add_user_to_group(user1['id'], group1['id'])
        PROVIDERS.identity_api.add_user_to_group(user1['id'], group2['id'])
        PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=self.project_bar['id'], role_id=self.role_member['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group1['id'], project_id=project1['id'], role_id=self.role_admin['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group2['id'], project_id=project2['id'], role_id=self.role_admin['id'])
        user_projects = PROVIDERS.assignment_api.list_projects_for_user(user1['id'])
        self.assertEqual(3, len(user_projects))

    def test_create_grant_no_user(self):
        PROVIDERS.assignment_api.create_grant(self.role_other['id'], user_id=uuid.uuid4().hex, project_id=self.project_bar['id'])

    def test_create_grant_no_group(self):
        PROVIDERS.assignment_api.create_grant(self.role_other['id'], group_id=uuid.uuid4().hex, project_id=self.project_bar['id'])

    def test_delete_group_removes_role_assignments(self):

        def get_member_assignments():
            assignments = PROVIDERS.assignment_api.list_role_assignments()
            return [x for x in assignments if x['role_id'] == default_fixtures.MEMBER_ROLE_ID]
        orig_member_assignments = get_member_assignments()
        new_group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        new_group = PROVIDERS.identity_api.create_group(new_group)
        new_project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        PROVIDERS.resource_api.create_project(new_project['id'], new_project)
        PROVIDERS.assignment_api.create_grant(group_id=new_group['id'], project_id=new_project['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
        PROVIDERS.identity_api.delete_group(new_group['id'])
        member_assignments = get_member_assignments()
        self.assertThat(member_assignments, matchers.Equals(orig_member_assignments))

    def test_get_roles_for_groups_on_domain(self):
        """Test retrieving group domain roles.

        Test Plan:

        - Create a domain, three groups and three roles
        - Assign one an inherited and the others a non-inherited group role
          to the domain
        - Ensure that only the non-inherited roles are returned on the domain

        """
        domain1 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
        group_list = []
        group_id_list = []
        role_list = []
        for _ in range(3):
            group = unit.new_group_ref(domain_id=domain1['id'])
            group = PROVIDERS.identity_api.create_group(group)
            group_list.append(group)
            group_id_list.append(group['id'])
            role = unit.new_role_ref()
            PROVIDERS.role_api.create_role(role['id'], role)
            role_list.append(role)
        PROVIDERS.assignment_api.create_grant(group_id=group_list[0]['id'], domain_id=domain1['id'], role_id=role_list[0]['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group_list[1]['id'], domain_id=domain1['id'], role_id=role_list[1]['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group_list[2]['id'], domain_id=domain1['id'], role_id=role_list[2]['id'], inherited_to_projects=True)
        role_refs = PROVIDERS.assignment_api.get_roles_for_groups(group_id_list, domain_id=domain1['id'])
        self.assertThat(role_refs, matchers.HasLength(2))
        self.assertIn(role_list[0], role_refs)
        self.assertIn(role_list[1], role_refs)

    def test_get_roles_for_groups_on_project(self):
        """Test retrieving group project roles.

        Test Plan:

        - Create two domains, two projects, six groups and six roles
        - Project1 is in Domain1, Project2 is in Domain2
        - Domain2/Project2 are spoilers
        - Assign a different direct group role to each project as well
          as both an inherited and non-inherited role to each domain
        - Get the group roles for Project 1 - depending on whether we have
          enabled inheritance, we should either get back just the direct role
          or both the direct one plus the inherited domain role from Domain 1

        """
        domain1 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
        domain2 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
        project1 = unit.new_project_ref(domain_id=domain1['id'])
        PROVIDERS.resource_api.create_project(project1['id'], project1)
        project2 = unit.new_project_ref(domain_id=domain2['id'])
        PROVIDERS.resource_api.create_project(project2['id'], project2)
        group_list = []
        group_id_list = []
        role_list = []
        for _ in range(6):
            group = unit.new_group_ref(domain_id=domain1['id'])
            group = PROVIDERS.identity_api.create_group(group)
            group_list.append(group)
            group_id_list.append(group['id'])
            role = unit.new_role_ref()
            PROVIDERS.role_api.create_role(role['id'], role)
            role_list.append(role)
        PROVIDERS.assignment_api.create_grant(group_id=group_list[0]['id'], domain_id=domain1['id'], role_id=role_list[0]['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group_list[1]['id'], domain_id=domain1['id'], role_id=role_list[1]['id'], inherited_to_projects=True)
        PROVIDERS.assignment_api.create_grant(group_id=group_list[2]['id'], project_id=project1['id'], role_id=role_list[2]['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group_list[3]['id'], domain_id=domain2['id'], role_id=role_list[3]['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group_list[4]['id'], domain_id=domain2['id'], role_id=role_list[4]['id'], inherited_to_projects=True)
        PROVIDERS.assignment_api.create_grant(group_id=group_list[5]['id'], project_id=project2['id'], role_id=role_list[5]['id'])
        role_refs = PROVIDERS.assignment_api.get_roles_for_groups(group_id_list, project_id=project1['id'])
        self.assertThat(role_refs, matchers.HasLength(2))
        self.assertIn(role_list[1], role_refs)
        self.assertIn(role_list[2], role_refs)

    def test_list_domains_for_groups(self):
        """Test retrieving domains for a list of groups.

        Test Plan:

        - Create three domains, three groups and one role
        - Assign a non-inherited group role to two domains, and an inherited
          group role to the third
        - Ensure only the domains with non-inherited roles are returned

        """
        domain_list = []
        group_list = []
        group_id_list = []
        for _ in range(3):
            domain = unit.new_domain_ref()
            PROVIDERS.resource_api.create_domain(domain['id'], domain)
            domain_list.append(domain)
            group = unit.new_group_ref(domain_id=domain['id'])
            group = PROVIDERS.identity_api.create_group(group)
            group_list.append(group)
            group_id_list.append(group['id'])
        role1 = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role1['id'], role1)
        PROVIDERS.assignment_api.create_grant(group_id=group_list[0]['id'], domain_id=domain_list[0]['id'], role_id=role1['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group_list[1]['id'], domain_id=domain_list[1]['id'], role_id=role1['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group_list[2]['id'], domain_id=domain_list[2]['id'], role_id=role1['id'], inherited_to_projects=True)
        domain_refs = PROVIDERS.assignment_api.list_domains_for_groups(group_id_list)
        self.assertThat(domain_refs, matchers.HasLength(2))
        self.assertIn(domain_list[0], domain_refs)
        self.assertIn(domain_list[1], domain_refs)

    def test_list_projects_for_groups(self):
        """Test retrieving projects for a list of groups.

        Test Plan:

        - Create two domains, four projects, seven groups and seven roles
        - Project1-3 are in Domain1, Project4 is in Domain2
        - Domain2/Project4 are spoilers
        - Project1 and 2 have direct group roles, Project3 has no direct
          roles but should inherit a group role from Domain1
        - Get the projects for the group roles that are assigned to Project1
          Project2 and the inherited one on Domain1. Depending on whether we
          have enabled inheritance, we should either get back just the projects
          with direct roles (Project 1 and 2) or also Project3 due to its
          inherited role from Domain1.

        """
        domain1 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain1['id'], domain1)
        domain2 = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
        project1 = unit.new_project_ref(domain_id=domain1['id'])
        project1 = PROVIDERS.resource_api.create_project(project1['id'], project1)
        project2 = unit.new_project_ref(domain_id=domain1['id'])
        project2 = PROVIDERS.resource_api.create_project(project2['id'], project2)
        project3 = unit.new_project_ref(domain_id=domain1['id'])
        project3 = PROVIDERS.resource_api.create_project(project3['id'], project3)
        project4 = unit.new_project_ref(domain_id=domain2['id'])
        project4 = PROVIDERS.resource_api.create_project(project4['id'], project4)
        group_list = []
        role_list = []
        for _ in range(7):
            group = unit.new_group_ref(domain_id=domain1['id'])
            group = PROVIDERS.identity_api.create_group(group)
            group_list.append(group)
            role = unit.new_role_ref()
            PROVIDERS.role_api.create_role(role['id'], role)
            role_list.append(role)
        PROVIDERS.assignment_api.create_grant(group_id=group_list[0]['id'], domain_id=domain1['id'], role_id=role_list[0]['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group_list[1]['id'], domain_id=domain1['id'], role_id=role_list[1]['id'], inherited_to_projects=True)
        PROVIDERS.assignment_api.create_grant(group_id=group_list[2]['id'], project_id=project1['id'], role_id=role_list[2]['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group_list[3]['id'], project_id=project2['id'], role_id=role_list[3]['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group_list[4]['id'], domain_id=domain2['id'], role_id=role_list[4]['id'])
        PROVIDERS.assignment_api.create_grant(group_id=group_list[5]['id'], domain_id=domain2['id'], role_id=role_list[5]['id'], inherited_to_projects=True)
        PROVIDERS.assignment_api.create_grant(group_id=group_list[6]['id'], project_id=project4['id'], role_id=role_list[6]['id'])
        group_id_list = [group_list[1]['id'], group_list[2]['id'], group_list[3]['id']]
        project_refs = PROVIDERS.assignment_api.list_projects_for_groups(group_id_list)
        self.assertThat(project_refs, matchers.HasLength(3))
        self.assertIn(project1, project_refs)
        self.assertIn(project2, project_refs)
        self.assertIn(project3, project_refs)

    def test_update_role_no_name(self):
        PROVIDERS.role_api.update_role(self.role_member['id'], {'description': uuid.uuid4().hex})

    def test_update_role_same_name(self):
        PROVIDERS.role_api.update_role(self.role_member['id'], {'name': self.role_member['name']})

    def _test_list_role_assignment_containing_names(self, domain_role=False):
        new_domain = self._get_domain_fixture()
        if domain_role:
            new_role = unit.new_role_ref(domain_id=new_domain['id'])
        else:
            new_role = unit.new_role_ref()
        new_user = unit.new_user_ref(domain_id=new_domain['id'])
        new_project = unit.new_project_ref(domain_id=new_domain['id'])
        new_group = unit.new_group_ref(domain_id=new_domain['id'])
        new_role = PROVIDERS.role_api.create_role(new_role['id'], new_role)
        new_user = PROVIDERS.identity_api.create_user(new_user)
        new_group = PROVIDERS.identity_api.create_group(new_group)
        PROVIDERS.resource_api.create_project(new_project['id'], new_project)
        PROVIDERS.assignment_api.create_grant(user_id=new_user['id'], project_id=new_project['id'], role_id=new_role['id'])
        PROVIDERS.assignment_api.create_grant(group_id=new_group['id'], project_id=new_project['id'], role_id=new_role['id'])
        PROVIDERS.assignment_api.create_grant(domain_id=new_domain['id'], user_id=new_user['id'], role_id=new_role['id'])
        _asgmt_prj = PROVIDERS.assignment_api.list_role_assignments(user_id=new_user['id'], project_id=new_project['id'], include_names=True)
        _asgmt_grp = PROVIDERS.assignment_api.list_role_assignments(group_id=new_group['id'], project_id=new_project['id'], include_names=True)
        _asgmt_dmn = PROVIDERS.assignment_api.list_role_assignments(domain_id=new_domain['id'], user_id=new_user['id'], include_names=True)
        self.assertThat(_asgmt_prj, matchers.HasLength(1))
        self.assertThat(_asgmt_grp, matchers.HasLength(1))
        self.assertThat(_asgmt_dmn, matchers.HasLength(1))
        first_asgmt_prj = _asgmt_prj[0]
        first_asgmt_grp = _asgmt_grp[0]
        first_asgmt_dmn = _asgmt_dmn[0]
        self.assertEqual(new_project['name'], first_asgmt_prj['project_name'])
        self.assertEqual(new_project['domain_id'], first_asgmt_prj['project_domain_id'])
        self.assertEqual(new_user['name'], first_asgmt_prj['user_name'])
        self.assertEqual(new_user['domain_id'], first_asgmt_prj['user_domain_id'])
        self.assertEqual(new_role['name'], first_asgmt_prj['role_name'])
        if domain_role:
            self.assertEqual(new_role['domain_id'], first_asgmt_prj['role_domain_id'])
        self.assertEqual(new_group['name'], first_asgmt_grp['group_name'])
        self.assertEqual(new_group['domain_id'], first_asgmt_grp['group_domain_id'])
        self.assertEqual(new_project['name'], first_asgmt_grp['project_name'])
        self.assertEqual(new_project['domain_id'], first_asgmt_grp['project_domain_id'])
        self.assertEqual(new_role['name'], first_asgmt_grp['role_name'])
        if domain_role:
            self.assertEqual(new_role['domain_id'], first_asgmt_grp['role_domain_id'])
        self.assertEqual(new_domain['name'], first_asgmt_dmn['domain_name'])
        self.assertEqual(new_user['name'], first_asgmt_dmn['user_name'])
        self.assertEqual(new_user['domain_id'], first_asgmt_dmn['user_domain_id'])
        self.assertEqual(new_role['name'], first_asgmt_dmn['role_name'])
        if domain_role:
            self.assertEqual(new_role['domain_id'], first_asgmt_dmn['role_domain_id'])

    def test_list_role_assignment_containing_names_global_role(self):
        self._test_list_role_assignment_containing_names()

    def test_list_role_assignment_containing_names_domain_role(self):
        self._test_list_role_assignment_containing_names(domain_role=True)

    def test_list_role_assignment_does_not_contain_names(self):
        """Test names are not included with list role assignments.

        Scenario:
            - names are NOT included by default
            - names are NOT included when include_names=False

        """

        def assert_does_not_contain_names(assignment):
            first_asgmt_prj = assignment[0]
            self.assertNotIn('project_name', first_asgmt_prj)
            self.assertNotIn('project_domain_id', first_asgmt_prj)
            self.assertNotIn('user_name', first_asgmt_prj)
            self.assertNotIn('user_domain_id', first_asgmt_prj)
            self.assertNotIn('role_name', first_asgmt_prj)
            self.assertNotIn('role_domain_id', first_asgmt_prj)
        new_role = unit.new_role_ref()
        new_domain = self._get_domain_fixture()
        new_user = unit.new_user_ref(domain_id=new_domain['id'])
        new_project = unit.new_project_ref(domain_id=new_domain['id'])
        new_role = PROVIDERS.role_api.create_role(new_role['id'], new_role)
        new_user = PROVIDERS.identity_api.create_user(new_user)
        PROVIDERS.resource_api.create_project(new_project['id'], new_project)
        PROVIDERS.assignment_api.create_grant(user_id=new_user['id'], project_id=new_project['id'], role_id=new_role['id'])
        role_assign_without_names = PROVIDERS.assignment_api.list_role_assignments(user_id=new_user['id'], project_id=new_project['id'])
        assert_does_not_contain_names(role_assign_without_names)
        role_assign_without_names = PROVIDERS.assignment_api.list_role_assignments(user_id=new_user['id'], project_id=new_project['id'], include_names=False)
        assert_does_not_contain_names(role_assign_without_names)

    def test_delete_user_assignments_user_same_id_as_group(self):
        """Test deleting user assignments when user_id == group_id.

        In this scenario, only user assignments must be deleted (i.e.
        USER_DOMAIN or USER_PROJECT).

        Test plan:
        * Create a user and a group with the same ID;
        * Create four roles and assign them to both user and group;
        * Delete all user assignments;
        * Group assignments must stay intact.
        """
        common_id = uuid.uuid4().hex
        project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        project = PROVIDERS.resource_api.create_project(project['id'], project)
        user = unit.new_user_ref(id=common_id, domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.driver.create_user(common_id, user)
        self.assertEqual(common_id, user['id'])
        group = unit.new_group_ref(id=common_id, domain_id=CONF.identity.default_domain_id)
        group = PROVIDERS.identity_api.driver.create_group(common_id, group)
        self.assertEqual(common_id, group['id'])
        roles = []
        for _ in range(4):
            role = unit.new_role_ref()
            roles.append(PROVIDERS.role_api.create_role(role['id'], role))
        PROVIDERS.assignment_api.driver.create_grant(user_id=user['id'], domain_id=CONF.identity.default_domain_id, role_id=roles[0]['id'])
        PROVIDERS.assignment_api.driver.create_grant(user_id=user['id'], project_id=project['id'], role_id=roles[1]['id'])
        PROVIDERS.assignment_api.driver.create_grant(group_id=group['id'], domain_id=CONF.identity.default_domain_id, role_id=roles[2]['id'])
        PROVIDERS.assignment_api.driver.create_grant(group_id=group['id'], project_id=project['id'], role_id=roles[3]['id'])
        user_assignments = PROVIDERS.assignment_api.list_role_assignments(user_id=user['id'])
        self.assertThat(user_assignments, matchers.HasLength(2))
        group_assignments = PROVIDERS.assignment_api.list_role_assignments(group_id=group['id'])
        self.assertThat(group_assignments, matchers.HasLength(2))
        PROVIDERS.assignment_api.delete_user_assignments(user_id=user['id'])
        user_assignments = PROVIDERS.assignment_api.list_role_assignments(user_id=user['id'])
        self.assertThat(user_assignments, matchers.HasLength(0))
        group_assignments = PROVIDERS.assignment_api.list_role_assignments(group_id=group['id'])
        self.assertThat(group_assignments, matchers.HasLength(2))
        for assignment in group_assignments:
            self.assertThat(assignment.keys(), matchers.Contains('group_id'))

    def test_delete_group_assignments_group_same_id_as_user(self):
        """Test deleting group assignments when group_id == user_id.

        In this scenario, only group assignments must be deleted (i.e.
        GROUP_DOMAIN or GROUP_PROJECT).

        Test plan:
        * Create a group and a user with the same ID;
        * Create four roles and assign them to both group and user;
        * Delete all group assignments;
        * User assignments must stay intact.
        """
        common_id = uuid.uuid4().hex
        project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
        project = PROVIDERS.resource_api.create_project(project['id'], project)
        user = unit.new_user_ref(id=common_id, domain_id=CONF.identity.default_domain_id)
        user = PROVIDERS.identity_api.driver.create_user(common_id, user)
        self.assertEqual(common_id, user['id'])
        group = unit.new_group_ref(id=common_id, domain_id=CONF.identity.default_domain_id)
        group = PROVIDERS.identity_api.driver.create_group(common_id, group)
        self.assertEqual(common_id, group['id'])
        roles = []
        for _ in range(4):
            role = unit.new_role_ref()
            roles.append(PROVIDERS.role_api.create_role(role['id'], role))
        PROVIDERS.assignment_api.driver.create_grant(user_id=user['id'], domain_id=CONF.identity.default_domain_id, role_id=roles[0]['id'])
        PROVIDERS.assignment_api.driver.create_grant(user_id=user['id'], project_id=project['id'], role_id=roles[1]['id'])
        PROVIDERS.assignment_api.driver.create_grant(group_id=group['id'], domain_id=CONF.identity.default_domain_id, role_id=roles[2]['id'])
        PROVIDERS.assignment_api.driver.create_grant(group_id=group['id'], project_id=project['id'], role_id=roles[3]['id'])
        user_assignments = PROVIDERS.assignment_api.list_role_assignments(user_id=user['id'])
        self.assertThat(user_assignments, matchers.HasLength(2))
        group_assignments = PROVIDERS.assignment_api.list_role_assignments(group_id=group['id'])
        self.assertThat(group_assignments, matchers.HasLength(2))
        PROVIDERS.assignment_api.delete_group_assignments(group_id=group['id'])
        group_assignments = PROVIDERS.assignment_api.list_role_assignments(group_id=group['id'])
        self.assertThat(group_assignments, matchers.HasLength(0))
        user_assignments = PROVIDERS.assignment_api.list_role_assignments(user_id=user['id'])
        self.assertThat(user_assignments, matchers.HasLength(2))
        for assignment in group_assignments:
            self.assertThat(assignment.keys(), matchers.Contains('user_id'))

    def test_remove_foreign_assignments_when_deleting_a_domain(self):
        group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
        group = PROVIDERS.identity_api.create_group(group)
        role = unit.new_role_ref()
        role = PROVIDERS.role_api.create_role(role['id'], role)
        new_domains = [unit.new_domain_ref(), unit.new_domain_ref()]
        for new_domain in new_domains:
            PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
            PROVIDERS.assignment_api.create_grant(group_id=group['id'], domain_id=new_domain['id'], role_id=role['id'])
            PROVIDERS.assignment_api.create_grant(user_id=self.user_two['id'], domain_id=new_domain['id'], role_id=role['id'])
        role_assignments = PROVIDERS.assignment_api.list_role_assignments(role_id=role['id'])
        self.assertThat(role_assignments, matchers.HasLength(4))
        PROVIDERS.resource_api.update_domain(new_domains[0]['id'], {'enabled': False})
        PROVIDERS.resource_api.delete_domain(new_domains[0]['id'])
        role_assignments = PROVIDERS.assignment_api.list_role_assignments(role_id=role['id'])
        self.assertThat(role_assignments, matchers.HasLength(2))
        PROVIDERS.resource_api.update_domain(new_domains[1]['id'], {'enabled': False})
        PROVIDERS.resource_api.delete_domain(new_domains[1]['id'])
        role_assignments = PROVIDERS.assignment_api.list_role_assignments(role_id=role['id'])
        self.assertEqual([], role_assignments)