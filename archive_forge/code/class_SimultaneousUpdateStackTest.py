import copy
import json
import time
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
class SimultaneousUpdateStackTest(functional_base.FunctionalTestsBase):

    @test.requires_convergence
    def test_retrigger_success(self):
        before, after = get_templates()
        stack_id = self.stack_create(template=before, expected_status='CREATE_IN_PROGRESS')
        time.sleep(10)
        self.update_stack(stack_id, after)

    @test.requires_convergence
    def test_retrigger_failure(self):
        before, after = get_templates(fail=True)
        stack_id = self.stack_create(template=before, expected_status='CREATE_IN_PROGRESS')
        time.sleep(10)
        self.update_stack(stack_id, after)

    @test.requires_convergence
    def test_retrigger_timeout(self):
        before, after = get_templates(delay_s=70)
        stack_id = self.stack_create(template=before, expected_status='CREATE_IN_PROGRESS', timeout=1)
        time.sleep(50)
        self.update_stack(stack_id, after)