import copy
import eventlet
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def _test_cancel_update(self, rollback=True, expected_status='ROLLBACK_COMPLETE'):
    before, after = get_templates()
    stack_id = self.stack_create(template=before)
    self.update_stack(stack_id, template=after, expected_status='UPDATE_IN_PROGRESS')
    self._wait_for_resource_status(stack_id, 'test1', 'UPDATE_IN_PROGRESS')
    self.cancel_update_stack(stack_id, rollback, expected_status)
    return stack_id