from heat_integrationtests.functional import functional_base
class AdminActionsTest(functional_base.FunctionalTestsBase):

    def setUp(self):
        super(AdminActionsTest, self).setUp()
        if not self.conf.admin_username or not self.conf.admin_password:
            self.skipTest('No admin creds found, skipping')

    def create_stack_setup_admin_client(self, template=test_template):
        self.stack_identifier = self.stack_create(template=template)
        self.setup_clients_for_admin()

    def test_admin_simple_stack_actions(self):
        self.create_stack_setup_admin_client()
        updated_template = test_template.copy()
        props = updated_template['resources']['test1']['properties']
        props['value'] = 'new_value'
        self.update_stack(self.stack_identifier, template=updated_template)
        self.stack_suspend(self.stack_identifier)
        self.stack_resume(self.stack_identifier)
        initial_resources = {'test1': 'OS::Heat::TestResource'}
        self.assertEqual(initial_resources, self.list_resources(self.stack_identifier))
        self._stack_delete(self.stack_identifier)

    def test_admin_complex_stack_actions(self):
        self.create_stack_setup_admin_client(template=rsg_template)
        updated_template = rsg_template.copy()
        props = updated_template['resources']['random_group']['properties']
        props['count'] = 3
        self.update_stack(self.stack_identifier, template=updated_template)
        self.stack_suspend(self.stack_identifier)
        self.stack_resume(self.stack_identifier)
        resources = {'random_group': 'OS::Heat::ResourceGroup'}
        self.assertEqual(resources, self.list_resources(self.stack_identifier))
        self._stack_delete(self.stack_identifier)