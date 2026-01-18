import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def _test_conditional(self, test3_resource):
    """Update manages new conditions added.

        When a new resource is added during updates, the stacks handles the new
        conditions correctly, and doesn't fail to load them while the update is
        still in progress.
        """
    stack_identifier = self.stack_create(template=test_template_one_resource)
    updated_template = copy.deepcopy(test_template_two_resource)
    updated_template['conditions'] = {'cond1': True}
    updated_template['resources']['test3'] = test3_resource
    test2_props = updated_template['resources']['test2']['properties']
    test2_props['action_wait_secs'] = {'create': 30}
    self.update_stack(stack_identifier, template=updated_template, expected_status='UPDATE_IN_PROGRESS')

    def check_resources():

        def is_complete(r):
            return r.resource_status in {'CREATE_COMPLETE', 'UPDATE_COMPLETE'}
        resources = self.list_resources(stack_identifier, is_complete)
        if len(resources) < 2:
            return False
        self.assertIn('test3', resources)
        return True
    self.assertTrue(test.call_until_true(20, 2, check_resources))