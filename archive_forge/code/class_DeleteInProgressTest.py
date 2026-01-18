import time
from heat_integrationtests.functional import functional_base
class DeleteInProgressTest(functional_base.FunctionalTestsBase):
    root_template = '\nheat_template_version: 2013-05-23\nresources:\n    rg:\n        type: OS::Heat::ResourceGroup\n        properties:\n            count: 125\n            resource_def:\n                type: empty.yaml\n'
    empty_template = '\nheat_template_version: 2013-05-23\nresources:\n'

    def test_delete_nested_stacks_create_in_progress(self):
        files = {'empty.yaml': self.empty_template}
        identifier = self.stack_create(template=self.root_template, files=files, expected_status='CREATE_IN_PROGRESS')
        time.sleep(20)
        self._stack_delete(identifier)