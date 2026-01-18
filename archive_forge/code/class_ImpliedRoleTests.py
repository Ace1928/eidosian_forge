from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
class ImpliedRoleTests(AssignmentTestHelperMixin):

    def test_implied_role_crd(self):
        prior_role_ref = unit.new_role_ref()
        PROVIDERS.role_api.create_role(prior_role_ref['id'], prior_role_ref)
        implied_role_ref = unit.new_role_ref()
        PROVIDERS.role_api.create_role(implied_role_ref['id'], implied_role_ref)
        PROVIDERS.role_api.create_implied_role(prior_role_ref['id'], implied_role_ref['id'])
        implied_role = PROVIDERS.role_api.get_implied_role(prior_role_ref['id'], implied_role_ref['id'])
        expected_implied_role_ref = {'prior_role_id': prior_role_ref['id'], 'implied_role_id': implied_role_ref['id']}
        self.assertLessEqual(expected_implied_role_ref.items(), implied_role.items())
        PROVIDERS.role_api.delete_implied_role(prior_role_ref['id'], implied_role_ref['id'])
        self.assertRaises(exception.ImpliedRoleNotFound, PROVIDERS.role_api.get_implied_role, uuid.uuid4().hex, uuid.uuid4().hex)

    def test_delete_implied_role_returns_not_found(self):
        self.assertRaises(exception.ImpliedRoleNotFound, PROVIDERS.role_api.delete_implied_role, uuid.uuid4().hex, uuid.uuid4().hex)

    def test_role_assignments_simple_tree_of_implied_roles(self):
        """Test that implied roles are expanded out."""
        test_plan = {'entities': {'domains': {'users': 1, 'projects': 1}, 'roles': 4}, 'implied_roles': [{'role': 0, 'implied_roles': 1}, {'role': 1, 'implied_roles': [2, 3]}], 'assignments': [{'user': 0, 'role': 0, 'project': 0}], 'tests': [{'params': {'user': 0}, 'results': [{'user': 0, 'role': 0, 'project': 0}]}, {'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 1, 'project': 0, 'indirect': {'role': 0}}, {'user': 0, 'role': 2, 'project': 0, 'indirect': {'role': 1}}, {'user': 0, 'role': 3, 'project': 0, 'indirect': {'role': 1}}]}]}
        self.execute_assignment_plan(test_plan)

    def test_circular_inferences(self):
        """Test that implied roles are expanded out."""
        test_plan = {'entities': {'domains': {'users': 1, 'projects': 1}, 'roles': 4}, 'implied_roles': [{'role': 0, 'implied_roles': [1]}, {'role': 1, 'implied_roles': [2, 3]}, {'role': 3, 'implied_roles': [0]}], 'assignments': [{'user': 0, 'role': 0, 'project': 0}], 'tests': [{'params': {'user': 0}, 'results': [{'user': 0, 'role': 0, 'project': 0}]}, {'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 0, 'project': 0, 'indirect': {'role': 3}}, {'user': 0, 'role': 1, 'project': 0, 'indirect': {'role': 0}}, {'user': 0, 'role': 2, 'project': 0, 'indirect': {'role': 1}}, {'user': 0, 'role': 3, 'project': 0, 'indirect': {'role': 1}}]}]}
        self.execute_assignment_plan(test_plan)

    def test_role_assignments_directed_graph_of_implied_roles(self):
        """Test that a role can have multiple, different prior roles."""
        test_plan = {'entities': {'domains': {'users': 1, 'projects': 1}, 'roles': 6}, 'implied_roles': [{'role': 0, 'implied_roles': [1, 2]}, {'role': 1, 'implied_roles': [3, 4]}, {'role': 5, 'implied_roles': 4}], 'assignments': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 5, 'project': 0}], 'tests': [{'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 5, 'project': 0}, {'user': 0, 'role': 1, 'project': 0, 'indirect': {'role': 0}}, {'user': 0, 'role': 2, 'project': 0, 'indirect': {'role': 0}}, {'user': 0, 'role': 3, 'project': 0, 'indirect': {'role': 1}}, {'user': 0, 'role': 4, 'project': 0, 'indirect': {'role': 1}}, {'user': 0, 'role': 4, 'project': 0, 'indirect': {'role': 5}}]}]}
        test_data = self.execute_assignment_plan(test_plan)
        role_ids = PROVIDERS.assignment_api.get_roles_for_user_and_project(test_data['users'][0]['id'], test_data['projects'][0]['id'])
        self.assertThat(role_ids, matchers.HasLength(6))
        for x in range(0, 5):
            self.assertIn(test_data['roles'][x]['id'], role_ids)

    def test_role_assignments_implied_roles_filtered_by_role(self):
        """Test that you can filter by role even if roles are implied."""
        test_plan = {'entities': {'domains': {'users': 1, 'projects': 2}, 'roles': 4}, 'implied_roles': [{'role': 0, 'implied_roles': 1}, {'role': 1, 'implied_roles': [2, 3]}], 'assignments': [{'user': 0, 'role': 0, 'project': 0}, {'user': 0, 'role': 3, 'project': 1}], 'tests': [{'params': {'role': 3, 'effective': True}, 'results': [{'user': 0, 'role': 3, 'project': 0, 'indirect': {'role': 1}}, {'user': 0, 'role': 3, 'project': 1}]}]}
        self.execute_assignment_plan(test_plan)

    def test_role_assignments_simple_tree_of_implied_roles_on_domain(self):
        """Test that implied roles are expanded out when placed on a domain."""
        test_plan = {'entities': {'domains': {'users': 1}, 'roles': 4}, 'implied_roles': [{'role': 0, 'implied_roles': 1}, {'role': 1, 'implied_roles': [2, 3]}], 'assignments': [{'user': 0, 'role': 0, 'domain': 0}], 'tests': [{'params': {'user': 0}, 'results': [{'user': 0, 'role': 0, 'domain': 0}]}, {'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'domain': 0}, {'user': 0, 'role': 1, 'domain': 0, 'indirect': {'role': 0}}, {'user': 0, 'role': 2, 'domain': 0, 'indirect': {'role': 1}}, {'user': 0, 'role': 3, 'domain': 0, 'indirect': {'role': 1}}]}]}
        self.execute_assignment_plan(test_plan)

    def test_role_assignments_inherited_implied_roles(self):
        """Test that you can intermix inherited and implied roles."""
        test_plan = {'entities': {'domains': {'users': 1, 'projects': 1}, 'roles': 4}, 'implied_roles': [{'role': 0, 'implied_roles': 1}], 'assignments': [{'user': 0, 'role': 0, 'domain': 0, 'inherited_to_projects': True}], 'tests': [{'params': {'user': 0}, 'results': [{'user': 0, 'role': 0, 'domain': 0, 'inherited_to_projects': 'projects'}]}, {'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 0, 'project': 0, 'indirect': {'domain': 0}}, {'user': 0, 'role': 1, 'project': 0, 'indirect': {'domain': 0, 'role': 0}}]}]}
        self.execute_assignment_plan(test_plan)

    def test_role_assignments_domain_specific_with_implied_roles(self):
        test_plan = {'entities': {'domains': {'users': 1, 'projects': 1, 'roles': 2}, 'roles': 2}, 'implied_roles': [{'role': 0, 'implied_roles': [1]}, {'role': 1, 'implied_roles': [2, 3]}], 'assignments': [{'user': 0, 'role': 0, 'project': 0}], 'tests': [{'params': {'user': 0}, 'results': [{'user': 0, 'role': 0, 'project': 0}]}, {'params': {'user': 0, 'effective': True}, 'results': [{'user': 0, 'role': 2, 'project': 0, 'indirect': {'role': 1}}, {'user': 0, 'role': 3, 'project': 0, 'indirect': {'role': 1}}]}]}
        self.execute_assignment_plan(test_plan)