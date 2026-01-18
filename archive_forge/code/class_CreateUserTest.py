from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
class CreateUserTest(functional_base.FunctionalTestsBase):

    def get_user_and_project_outputs(self, stack_identifier):
        stack = self.client.stacks.get(stack_identifier)
        project_name = self._stack_output(stack, 'project_name')
        user_name = self._stack_output(stack, 'user_name')
        return (project_name, user_name)

    def get_outputs(self, stack_identifier, output_key):
        stack = self.client.stacks.get(stack_identifier)
        return self._stack_output(stack, output_key)

    def test_assign_user_role_with_domain(self):
        self.setup_clients_for_admin()
        parms = {'user_name': test.rand_name('test-user-domain-user-name'), 'project_name': test.rand_name('test-user-domain-project'), 'domain_name': test.rand_name('test-user-domain-domain-name')}
        stack_identifier_create_user = self.stack_create(template=create_user, parameters=parms)
        self.stack_create(template=assign_user_roles, parameters=parms)
        project_name, user_name = self.get_user_and_project_outputs(stack_identifier_create_user)
        self.assertEqual(project_name, project_name)
        self.assertEqual(user_name, user_name)
        users = self.keystone_client.users.list()
        projects = self.keystone_client.projects.list()
        user_id = [x for x in users if x.name == user_name][0].id
        project_id = [x for x in projects if x.name == project_name][0].id
        self.assertIsNotNone(self.keystone_client.role_assignments.list(user=user_id, project=project_id))
        self.update_stack(stack_identifier=stack_identifier_create_user, template=disable_domain, parameters=parms)