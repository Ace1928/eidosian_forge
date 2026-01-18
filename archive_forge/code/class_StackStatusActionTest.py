from unittest import mock
import testscenarios
from testscenarios import scenarios as scnrs
import testtools
from heatclient.v1 import stacks
class StackStatusActionTest(testtools.TestCase):
    scenarios = scnrs.multiply_scenarios([('CREATE', dict(action='CREATE')), ('DELETE', dict(action='DELETE')), ('UPDATE', dict(action='UPDATE')), ('ROLLBACK', dict(action='ROLLBACK')), ('SUSPEND', dict(action='SUSPEND')), ('RESUME', dict(action='RESUME')), ('CHECK', dict(action='CHECK'))], [('IN_PROGRESS', dict(status='IN_PROGRESS')), ('FAILED', dict(status='FAILED')), ('COMPLETE', dict(status='COMPLETE'))])

    def test_status_action(self):
        stack_status = '%s_%s' % (self.action, self.status)
        stack = mock_stack(None, 'stack_1', 'abcd1234')
        stack.stack_status = stack_status
        self.assertEqual(self.action, stack.action)
        self.assertEqual(self.status, stack.status)