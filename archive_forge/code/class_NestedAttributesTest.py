import json
from heatclient import exc as heat_exceptions
import yaml
from heat_integrationtests.functional import functional_base
class NestedAttributesTest(functional_base.FunctionalTestsBase):
    """Prove that we can use the template resource references."""
    main_templ = '\nheat_template_version: 2014-10-16\nresources:\n  secret2:\n    type: My::NestedSecret\noutputs:\n  old_way:\n    value: { get_attr: [secret2, nested_str]}\n  test_attr1:\n    value: { get_attr: [secret2, resource.secret1, value]}\n  test_attr2:\n    value: { get_attr: [secret2, resource.secret1.value]}\n  test_ref:\n    value: { get_resource: secret2 }\n'
    env_templ = '\nresource_registry:\n  "My::NestedSecret": nested.yaml\n'

    def test_stack_ref(self):
        nested_templ = '\nheat_template_version: 2014-10-16\nresources:\n  secret1:\n    type: OS::Heat::RandomString\noutputs:\n  nested_str:\n    value: {get_attr: [secret1, value]}\n'
        stack_identifier = self.stack_create(template=self.main_templ, files={'nested.yaml': nested_templ}, environment=self.env_templ)
        self.assert_resource_is_a_stack(stack_identifier, 'secret2')
        stack = self.client.stacks.get(stack_identifier)
        test_ref = self._stack_output(stack, 'test_ref')
        self.assertIn('arn:openstack:heat:', test_ref)

    def test_transparent_ref(self):
        """Test using nested resource more transparently.

        With the addition of OS::stack_id we can now use the nested resource
        more transparently.
        """
        nested_templ = '\nheat_template_version: 2014-10-16\nresources:\n  secret1:\n    type: OS::Heat::RandomString\noutputs:\n  OS::stack_id:\n    value: {get_resource: secret1}\n  nested_str:\n    value: {get_attr: [secret1, value]}\n'
        stack_identifier = self.stack_create(template=self.main_templ, files={'nested.yaml': nested_templ}, environment=self.env_templ)
        self.assert_resource_is_a_stack(stack_identifier, 'secret2')
        stack = self.client.stacks.get(stack_identifier)
        test_ref = self._stack_output(stack, 'test_ref')
        test_attr = self._stack_output(stack, 'old_way')
        self.assertNotIn('arn:openstack:heat', test_ref)
        self.assertEqual(test_attr, test_ref)

    def test_nested_attributes(self):
        nested_templ = '\nheat_template_version: 2014-10-16\nresources:\n  secret1:\n    type: OS::Heat::RandomString\noutputs:\n  nested_str:\n    value: {get_attr: [secret1, value]}\n'
        stack_identifier = self.stack_create(template=self.main_templ, files={'nested.yaml': nested_templ}, environment=self.env_templ)
        self.assert_resource_is_a_stack(stack_identifier, 'secret2')
        stack = self.client.stacks.get(stack_identifier)
        old_way = self._stack_output(stack, 'old_way')
        test_attr1 = self._stack_output(stack, 'test_attr1')
        test_attr2 = self._stack_output(stack, 'test_attr2')
        self.assertEqual(old_way, test_attr1)
        self.assertEqual(old_way, test_attr2)