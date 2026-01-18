import copy
import json
from heatclient import exc
import yaml
from heat_integrationtests.functional import functional_base
class ResourceGroupTestNullParams(functional_base.FunctionalTestsBase):
    template = '\nheat_template_version: 2013-05-23\nparameters:\n  param:\n    type: empty\nresources:\n  random_group:\n    type: OS::Heat::ResourceGroup\n    properties:\n      count: 1\n      resource_def:\n        type: My::RandomString\n        properties:\n          param: {get_param: param}\noutputs:\n  val:\n    value: {get_attr: [random_group, val]}\n'
    nested_template_file = '\nheat_template_version: 2013-05-23\nparameters:\n  param:\n    type: empty\noutputs:\n  val:\n    value: {get_param: param}\n'
    scenarios = [('string_empty', dict(param='', p_type='string')), ('boolean_false', dict(param=False, p_type='boolean')), ('number_zero', dict(param=0, p_type='number')), ('comma_delimited_list', dict(param=[], p_type='comma_delimited_list')), ('json_empty', dict(param={}, p_type='json'))]

    def test_create_pass_zero_parameter(self):
        templ = self.template.replace('type: empty', 'type: %s' % self.p_type)
        n_t_f = self.nested_template_file.replace('type: empty', 'type: %s' % self.p_type)
        files = {'provider.yaml': n_t_f}
        env = {'resource_registry': {'My::RandomString': 'provider.yaml'}}
        stack_identifier = self.stack_create(template=templ, files=files, environment=env, parameters={'param': self.param})
        stack = self.client.stacks.get(stack_identifier)
        self.assertEqual(self.param, self._stack_output(stack, 'val')[0])