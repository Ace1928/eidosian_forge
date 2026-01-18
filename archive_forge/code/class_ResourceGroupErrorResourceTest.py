import copy
import json
from heatclient import exc
import yaml
from heat_integrationtests.functional import functional_base
class ResourceGroupErrorResourceTest(functional_base.FunctionalTestsBase):
    template = '\nheat_template_version: "2013-05-23"\nresources:\n  group1:\n    type: OS::Heat::ResourceGroup\n    properties:\n      count: 2\n      resource_def:\n        type: fail.yaml\n'
    nested_templ = '\nheat_template_version: "2013-05-23"\nresources:\n  oops:\n    type: OS::Heat::TestResource\n    properties:\n      fail: true\n      wait_secs: 2\n'

    def test_fail(self):
        stack_identifier = self.stack_create(template=self.template, files={'fail.yaml': self.nested_templ}, expected_status='CREATE_FAILED', enable_cleanup=False)
        stack = self.client.stacks.get(stack_identifier)
        self.assertEqual('CREATE_FAILED', stack.stack_status)
        self.client.stacks.delete(stack_identifier)
        self._wait_for_stack_status(stack_identifier, 'DELETE_COMPLETE', success_on_not_found=True)