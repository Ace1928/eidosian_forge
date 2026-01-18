from heat_integrationtests.functional import functional_base
class ResourceChainTests(functional_base.FunctionalTestsBase):

    def test_create(self):
        params = {'string-length': 8}
        stack_id = self.stack_create(template=TEMPLATE_SIMPLE, parameters=params)
        stack = self.client.stacks.get(stack_id)
        self.assertIsNotNone(stack)
        expected = {'my-chain': 'OS::Heat::ResourceChain'}
        found = self.list_resources(stack_id)
        self.assertEqual(expected, found)
        nested_id = self.group_nested_identifier(stack_id, 'my-chain')
        expected = {'0': 'OS::Heat::RandomString', '1': 'OS::Heat::RandomString'}
        found = self.list_resources(nested_id)
        self.assertEqual(expected, found)
        resource_ids = self._stack_output(stack, 'resource-ids')
        self.assertIsNotNone(resource_ids)
        self.assertEqual(2, len(resource_ids))
        resource_value = self._stack_output(stack, 'resource-0-value')
        self.assertIsNotNone(resource_value)
        self.assertEqual(8, len(resource_value))
        resource_attrs = self._stack_output(stack, 'all-resource-attrs')
        self.assertIsNotNone(resource_attrs)
        self.assertIsInstance(resource_attrs, dict)
        self.assertEqual(2, len(resource_attrs))
        self.assertEqual(8, len(resource_attrs['0']))
        self.assertEqual(8, len(resource_attrs['1']))

    def test_update(self):
        params = {'string-length': 8}
        stack_id = self.stack_create(template=TEMPLATE_SIMPLE, parameters=params)
        update_tmpl = "\n        heat_template_version: 2016-04-08\n        parameters:\n          string-length:\n            type: number\n        resources:\n          my-chain:\n            type: OS::Heat::ResourceChain\n            properties:\n              resources: ['OS::Heat::None']\n        "
        self.update_stack(stack_id, template=update_tmpl, parameters=params)
        nested_id = self.group_nested_identifier(stack_id, 'my-chain')
        expected = {'0': 'OS::Heat::None'}
        found = self.list_resources(nested_id)
        self.assertEqual(expected, found)

    def test_update_resources(self):
        params = {'chain-types': 'OS::Heat::None'}
        stack_id = self.stack_create(template=TEMPLATE_PARAM_DRIVEN, parameters=params)
        nested_id = self.group_nested_identifier(stack_id, 'my-chain')
        expected = {'0': 'OS::Heat::None'}
        found = self.list_resources(nested_id)
        self.assertEqual(expected, found)
        params = {'chain-types': 'OS::Heat::None,OS::Heat::None'}
        self.update_stack(stack_id, template=TEMPLATE_PARAM_DRIVEN, parameters=params)
        expected = {'0': 'OS::Heat::None', '1': 'OS::Heat::None'}
        found = self.list_resources(nested_id)
        self.assertEqual(expected, found)

    def test_resources_param_driven(self):
        params = {'chain-types': 'OS::Heat::None,OS::Heat::RandomString,OS::Heat::None'}
        stack_id = self.stack_create(template=TEMPLATE_PARAM_DRIVEN, parameters=params)
        nested_id = self.group_nested_identifier(stack_id, 'my-chain')
        expected = {'0': 'OS::Heat::None', '1': 'OS::Heat::RandomString', '2': 'OS::Heat::None'}
        found = self.list_resources(nested_id)
        self.assertEqual(expected, found)

    def test_resources_env_defined(self):
        env = {'parameters': {'chain-types': 'OS::Heat::None'}}
        stack_id = self.stack_create(template=TEMPLATE_PARAM_DRIVEN, environment=env)
        nested_id = self.group_nested_identifier(stack_id, 'my-chain')
        expected = {'0': 'OS::Heat::None'}
        found = self.list_resources(nested_id)
        self.assertEqual(expected, found)