from heat_integrationtests.functional import functional_base
from heatclient import exc as heat_exceptions
class ImmutableParametersTest(functional_base.FunctionalTestsBase):
    template_param_has_no_immutable_field = "\nheat_template_version: 2014-10-16\nparameters:\n  param1:\n    type: string\n    default: default_value\noutputs:\n  param1_output:\n    description: 'parameter 1 details'\n    value: { get_param: param1 }\n"
    template_param_has_immutable_field = "\nheat_template_version: 2014-10-16\nparameters:\n  param1:\n    type: string\n    default: default_value\n    immutable: false\noutputs:\n  param1_output:\n    description: 'parameter 1 details'\n    value: { get_param: param1 }\n"

    def test_no_immutable_param_field(self):
        param1_create_value = 'value1'
        create_parameters = {'param1': param1_create_value}
        stack_identifier = self.stack_create(template=self.template_param_has_no_immutable_field, parameters=create_parameters)
        stack = self.client.stacks.get(stack_identifier)
        self.assertEqual(param1_create_value, self._stack_output(stack, 'param1_output'))
        param1_update_value = 'value2'
        update_parameters = {'param1': param1_update_value}
        self.update_stack(stack_identifier, template=self.template_param_has_no_immutable_field, parameters=update_parameters)
        stack = self.client.stacks.get(stack_identifier)
        self.assertEqual(param1_update_value, self._stack_output(stack, 'param1_output'))

    def test_immutable_param_field_allowed(self):
        param1_create_value = 'value1'
        create_parameters = {'param1': param1_create_value}
        stack_identifier = self.stack_create(template=self.template_param_has_immutable_field, parameters=create_parameters)
        stack = self.client.stacks.get(stack_identifier)
        self.assertEqual(param1_create_value, self._stack_output(stack, 'param1_output'))
        param1_update_value = 'value2'
        update_parameters = {'param1': param1_update_value}
        self.update_stack(stack_identifier, template=self.template_param_has_immutable_field, parameters=update_parameters)
        stack = self.client.stacks.get(stack_identifier)
        self.assertEqual(param1_update_value, self._stack_output(stack, 'param1_output'))
        self.assertEqual('UPDATE_COMPLETE', stack.stack_status)

    def test_immutable_param_field_error(self):
        param1_create_value = 'value1'
        create_parameters = {'param1': param1_create_value}
        immutable_true = self.template_param_has_immutable_field.replace('immutable: false', 'immutable: true')
        stack_identifier = self.stack_create(template=immutable_true, parameters=create_parameters)
        stack = self.client.stacks.get(stack_identifier)
        param1_update_value = 'value2'
        update_parameters = {'param1': param1_update_value}
        self.assertEqual(param1_create_value, self._stack_output(stack, 'param1_output'))
        try:
            self.update_stack(stack_identifier, template=immutable_true, parameters=update_parameters)
        except heat_exceptions.HTTPBadRequest as exc:
            exp = 'The following parameters are immutable and may not be updated: param1'
            self.assertIn(exp, str(exc))
        stack = self.client.stacks.get(stack_identifier)
        self.assertEqual('CREATE_COMPLETE', stack.stack_status)
        self.assertEqual(param1_create_value, self._stack_output(stack, 'param1_output'))