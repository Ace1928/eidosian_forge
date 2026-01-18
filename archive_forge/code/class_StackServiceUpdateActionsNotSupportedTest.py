from unittest import mock
from oslo_messaging.rpc import dispatcher
from heat.common import exception
from heat.common import template_format
from heat.engine import service
from heat.engine import stack
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
class StackServiceUpdateActionsNotSupportedTest(common.HeatTestCase):
    scenarios = [('suspend_in_progress', dict(action='SUSPEND', status='IN_PROGRESS')), ('suspend_complete', dict(action='SUSPEND', status='COMPLETE')), ('suspend_failed', dict(action='SUSPEND', status='FAILED')), ('delete_in_progress', dict(action='DELETE', status='IN_PROGRESS')), ('delete_complete', dict(action='DELETE', status='COMPLETE')), ('delete_failed', dict(action='DELETE', status='FAILED'))]

    def setUp(self):
        super(StackServiceUpdateActionsNotSupportedTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.man = service.EngineService('a-host', 'a-topic')

    @mock.patch.object(stack.Stack, 'load')
    def test_stack_update_actions_not_supported(self, mock_load):
        stack_name = '%s-%s' % (self.action, self.status)
        t = template_format.parse(tools.wp_template)
        old_stack = utils.parse_stack(t, stack_name=stack_name)
        old_stack.action = self.action
        old_stack.status = self.status
        sid = old_stack.store()
        s = stack_object.Stack.get_by_id(self.ctx, sid)
        mock_load.return_value = old_stack
        params = {'foo': 'bar'}
        template = '{ "Resources": {} }'
        ex = self.assertRaises(dispatcher.ExpectedException, self.man.update_stack, self.ctx, old_stack.identifier(), template, params, None, {})
        self.assertEqual(exception.NotSupported, ex.exc_info[0])
        mock_load.assert_called_once_with(self.ctx, stack=s, check_refresh_cred=True)
        old_stack.delete()