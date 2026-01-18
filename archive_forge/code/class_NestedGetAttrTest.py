from heat_integrationtests.functional import functional_base
class NestedGetAttrTest(functional_base.FunctionalTestsBase):

    def assertOutput(self, value, stack_identifier, key):
        op = self.client.stacks.output_show(stack_identifier, key)['output']
        self.assertEqual(key, op['output_key'])
        if 'output_error' in op:
            raise Exception(op['output_error'])
        self.assertEqual(value, op['output_value'])

    def test_nested_get_attr_create(self):
        stack_identifier = self.stack_create(template=initial_template)
        self.assertOutput('wibble', stack_identifier, 'value1')
        self.assertOutput('bar', stack_identifier, 'value2')
        self.assertOutput('barney', stack_identifier, 'value3')

    def test_nested_get_attr_update(self):
        stack_identifier = self.stack_create(template=initial_template)
        self.update_stack(stack_identifier, template=update_template)
        self.assertOutput('bar', stack_identifier, 'value1')
        self.assertOutput('barney', stack_identifier, 'value2')
        self.assertOutput('wibble', stack_identifier, 'value3')
        self.assertOutput('quux', stack_identifier, 'value4')