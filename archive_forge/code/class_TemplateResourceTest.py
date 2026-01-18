import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
class TemplateResourceTest(functional_base.FunctionalTestsBase):
    """Prove that we can use the registry in a nested provider."""
    template = '\nheat_template_version: 2013-05-23\nresources:\n  secret1:\n    type: OS::Heat::RandomString\noutputs:\n  secret-out:\n    value: { get_attr: [secret1, value] }\n'
    nested_templ = '\nheat_template_version: 2013-05-23\nresources:\n  secret2:\n    type: OS::Heat::RandomString\noutputs:\n  value:\n    value: { get_attr: [secret2, value] }\n'
    env_templ = '\nresource_registry:\n  "OS::Heat::RandomString": nested.yaml\n'

    def test_nested_env(self):
        main_templ = '\nheat_template_version: 2013-05-23\nresources:\n  secret1:\n    type: My::NestedSecret\noutputs:\n  secret-out:\n    value: { get_attr: [secret1, value] }\n'
        nested_templ = '\nheat_template_version: 2013-05-23\nresources:\n  secret2:\n    type: My::Secret\noutputs:\n  value:\n    value: { get_attr: [secret2, value] }\n'
        env_templ = '\nresource_registry:\n  "My::Secret": "OS::Heat::RandomString"\n  "My::NestedSecret": nested.yaml\n'
        stack_identifier = self.stack_create(template=main_templ, files={'nested.yaml': nested_templ}, environment=env_templ)
        nested_ident = self.assert_resource_is_a_stack(stack_identifier, 'secret1')
        sec2 = self.client.resources.get(nested_ident, 'secret2')
        self.assertEqual('secret1', sec2.parent_resource)

    def test_no_infinite_recursion(self):
        """Prove that we can override a python resource.

        And use that resource within the template resource.
        """
        stack_identifier = self.stack_create(template=self.template, files={'nested.yaml': self.nested_templ}, environment=self.env_templ)
        self.assert_resource_is_a_stack(stack_identifier, 'secret1')

    def test_nested_stack_delete_then_delete_parent_stack(self):
        """Check the robustness of stack deletion.

        This tests that if you manually delete a nested
        stack, the parent stack is still deletable.
        """
        stack_identifier = self.stack_create(template=self.template, files={'nested.yaml': self.nested_templ}, environment=self.env_templ, enable_cleanup=False)
        nested_ident = self.assert_resource_is_a_stack(stack_identifier, 'secret1')
        self._stack_delete(nested_ident)
        self._stack_delete(stack_identifier)

    def test_change_in_file_path(self):
        stack_identifier = self.stack_create(template=self.template, files={'nested.yaml': self.nested_templ}, environment=self.env_templ)
        stack = self.client.stacks.get(stack_identifier)
        secret_out1 = self._stack_output(stack, 'secret-out')
        nested_templ_2 = '\nheat_template_version: 2013-05-23\nresources:\n  secret2:\n    type: OS::Heat::RandomString\noutputs:\n  value:\n    value: freddy\n'
        env_templ_2 = '\nresource_registry:\n  "OS::Heat::RandomString": new/nested.yaml\n'
        self.update_stack(stack_identifier, template=self.template, files={'new/nested.yaml': nested_templ_2}, environment=env_templ_2)
        stack = self.client.stacks.get(stack_identifier)
        secret_out2 = self._stack_output(stack, 'secret-out')
        self.assertNotEqual(secret_out1, secret_out2)
        self.assertEqual('freddy', secret_out2)