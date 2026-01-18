from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from oslo_serialization import jsonutils as json
from heat.common import context
from heat.common import environment_util as env_util
from heat.common import exception
from heat.common import identifier
from heat.common import policy
from heat.common import template_format
from heat.engine.cfn import template as cfntemplate
from heat.engine import environment
from heat.engine.hot import functions as hot_functions
from heat.engine.hot import template as hottemplate
from heat.engine import resource as res
from heat.engine import service
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class StackConvergenceServiceCreateUpdateTest(common.HeatTestCase):

    def setUp(self):
        super(StackConvergenceServiceCreateUpdateTest, self).setUp()
        cfg.CONF.set_override('convergence_engine', True)
        self.ctx = utils.dummy_context()
        self.man = service.EngineService('a-host', 'a-topic')
        self.man.thread_group_mgr = tools.DummyThreadGroupManager()

    def _stub_update_mocks(self, stack_to_return):
        self.patchobject(parser, 'Stack')
        parser.Stack.load.return_value = stack_to_return
        self.patchobject(templatem, 'Template')
        self.patchobject(environment, 'Environment')

    def _test_stack_create_convergence(self, stack_name):
        params = {'foo': 'bar'}
        template = '{ "Template": "data" }'
        stack = tools.get_stack(stack_name, self.ctx, template=tools.string_template_five, convergence=True)
        stack.converge = None
        self.patchobject(templatem, 'Template', return_value=stack.t)
        self.patchobject(environment, 'Environment', return_value=stack.env)
        self.patchobject(parser, 'Stack', return_value=stack)
        self.patchobject(stack, 'validate', return_value=None)
        api_args = {'timeout_mins': 60, 'disable_rollback': False}
        result = self.man.create_stack(self.ctx, 'service_create_test_stack', template, params, None, api_args)
        db_stack = stack_object.Stack.get_by_id(self.ctx, result['stack_id'])
        self.assertTrue(db_stack.convergence)
        self.assertEqual(result['stack_id'], db_stack.id)
        templatem.Template.assert_called_once_with(template, files=None)
        environment.Environment.assert_called_once_with(params)
        parser.Stack.assert_called_once_with(self.ctx, stack.name, stack.t, owner_id=None, parent_resource=None, nested_depth=0, user_creds_id=None, stack_user_project_id=None, timeout_mins=60, disable_rollback=False, convergence=True)

    def test_stack_create_enabled_convergence_engine(self):
        stack_name = 'service_create_test_stack'
        self._test_stack_create_convergence(stack_name)

    def test_stack_update_enabled_convergence_engine(self):
        stack_name = 'service_update_test_stack'
        params = {'foo': 'bar'}
        template = '{ "Template": "data" }'
        old_stack = tools.get_stack(stack_name, self.ctx, template=tools.string_template_five, convergence=True)
        old_stack.timeout_mins = 1
        old_stack.store()
        stack = tools.get_stack(stack_name, self.ctx, template=tools.string_template_five_update, convergence=True)
        self._stub_update_mocks(old_stack)
        templatem.Template.return_value = stack.t
        environment.Environment.return_value = stack.env
        parser.Stack.return_value = stack
        self.patchobject(stack, 'validate', return_value=None)
        api_args = {'timeout_mins': 60, 'disable_rollback': False, rpc_api.PARAM_CONVERGE: False}
        result = self.man.update_stack(self.ctx, old_stack.identifier(), template, params, None, api_args)
        self.assertTrue(old_stack.convergence)
        self.assertEqual(old_stack.identifier(), result)
        self.assertIsInstance(result, dict)
        self.assertTrue(result['stack_id'])
        parser.Stack.load.assert_called_once_with(self.ctx, stack=mock.ANY, check_refresh_cred=True)
        templatem.Template.assert_called_once_with(template, files=None)
        environment.Environment.assert_called_once_with(params)