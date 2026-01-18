import copy
import json
from heatclient import exc
import yaml
from heat_integrationtests.functional import functional_base
class ResourceGroupAdoptTest(functional_base.FunctionalTestsBase):
    """Prove that we can do resource group adopt."""
    main_template = '\nheat_template_version: "2013-05-23"\nresources:\n  group1:\n    type: OS::Heat::ResourceGroup\n    properties:\n      count: 2\n      resource_def:\n        type: OS::Heat::RandomString\noutputs:\n  test0:\n    value: {get_attr: [group1, resource.0.value]}\n  test1:\n    value: {get_attr: [group1, resource.1.value]}\n'

    def _yaml_to_json(self, yaml_templ):
        return yaml.safe_load(yaml_templ)

    def test_adopt(self):
        data = {'resources': {'group1': {'status': 'COMPLETE', 'name': 'group1', 'resource_data': {}, 'metadata': {}, 'resource_id': 'test-group1-id', 'action': 'CREATE', 'type': 'OS::Heat::ResourceGroup', 'resources': {'0': {'status': 'COMPLETE', 'name': '0', 'resource_data': {'value': 'goopie'}, 'resource_id': 'ID-0', 'action': 'CREATE', 'type': 'OS::Heat::RandomString', 'metadata': {}}, '1': {'status': 'COMPLETE', 'name': '1', 'resource_data': {'value': 'different'}, 'resource_id': 'ID-1', 'action': 'CREATE', 'type': 'OS::Heat::RandomString', 'metadata': {}}}}}, 'environment': {'parameters': {}}, 'template': yaml.safe_load(self.main_template)}
        stack_identifier = self.stack_adopt(adopt_data=json.dumps(data))
        self.assert_resource_is_a_stack(stack_identifier, 'group1')
        stack = self.client.stacks.get(stack_identifier)
        self.assertEqual('goopie', self._stack_output(stack, 'test0'))
        self.assertEqual('different', self._stack_output(stack, 'test1'))