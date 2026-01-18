from unittest import mock
import uuid
import eventlet
from oslo_config import cfg
from heat.common import exception
from heat.engine import check_resource
from heat.engine import dependencies
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import sync_point
from heat.engine import worker
from heat.rpc import api as rpc_api
from heat.rpc import worker_client
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
class MiscMethodsTest(common.HeatTestCase):

    def setUp(self):
        super(MiscMethodsTest, self).setUp()
        cfg.CONF.set_default('convergence_engine', True)
        self.ctx = utils.dummy_context()
        self.stack = tools.get_stack('check_workflow_create_stack', self.ctx, template=tools.attr_cache_template, convergence=True)
        self.stack.converge_stack(self.stack.t)
        self.resource = self.stack['A']

    def test_node_data_ok(self):
        self.resource.action = self.resource.CREATE
        expected_input_data = {'attrs': {(u'flat_dict', u'key2'): 'val2', (u'flat_dict', u'key3'): 'val3', (u'nested_dict', u'dict', u'a'): 1, (u'nested_dict', u'dict', u'b'): 2}, 'id': mock.ANY, 'reference_id': 'A', 'name': 'A', 'uuid': mock.ANY, 'action': mock.ANY, 'status': mock.ANY}
        actual_input_data = self.resource.node_data()
        self.assertEqual(expected_input_data, actual_input_data.as_dict())

    def test_node_data_exception(self):
        self.resource.action = self.resource.CREATE
        expected_input_data = {'attrs': {}, 'id': mock.ANY, 'reference_id': 'A', 'name': 'A', 'uuid': mock.ANY, 'action': mock.ANY, 'status': mock.ANY}
        self.resource.get_attribute = mock.Mock(side_effect=exception.InvalidTemplateAttribute(resource='A', key='value'))
        actual_input_data = self.resource.node_data()
        self.assertEqual(expected_input_data, actual_input_data.as_dict())

    @mock.patch.object(sync_point, 'sync')
    def test_check_stack_complete_root(self, mock_sync):
        check_resource.check_stack_complete(self.ctx, self.stack, self.stack.current_traversal, self.stack['E'].id, self.stack.convergence_dependencies, True)
        mock_sync.assert_called_once_with(self.ctx, self.stack.id, self.stack.current_traversal, True, mock.ANY, mock.ANY, {(self.stack['E'].id, True): None})

    @mock.patch.object(sync_point, 'sync')
    def test_check_stack_complete_child(self, mock_sync):
        check_resource.check_stack_complete(self.ctx, self.stack, self.stack.current_traversal, self.resource.id, self.stack.convergence_dependencies, True)
        self.assertFalse(mock_sync.called)

    @mock.patch.object(dependencies.Dependencies, 'roots')
    @mock.patch.object(stack.Stack, '_persist_state')
    def test_check_stack_complete_persist_called(self, mock_persist_state, mock_dep_roots):
        mock_dep_roots.return_value = [(1, True)]
        check_resource.check_stack_complete(self.ctx, self.stack, self.stack.current_traversal, 1, self.stack.convergence_dependencies, True)
        self.assertTrue(mock_persist_state.called)

    @mock.patch.object(sync_point, 'sync')
    def test_propagate_check_resource(self, mock_sync):
        check_resource.propagate_check_resource(self.ctx, mock.ANY, mock.ANY, self.stack.current_traversal, mock.ANY, ('A', True), {}, True, None)
        self.assertTrue(mock_sync.called)

    @mock.patch.object(resource.Resource, 'create_convergence')
    @mock.patch.object(resource.Resource, 'update_convergence')
    def test_check_resource_update_init_action(self, mock_update, mock_create):
        self.resource.action = 'INIT'
        check_resource.check_resource_update(self.resource, self.resource.stack.t.id, set(), 'engine-id', self.stack, None)
        self.assertTrue(mock_create.called)
        self.assertFalse(mock_update.called)

    @mock.patch.object(resource.Resource, 'create_convergence')
    @mock.patch.object(resource.Resource, 'update_convergence')
    def test_check_resource_update_create_action(self, mock_update, mock_create):
        self.resource.action = 'CREATE'
        check_resource.check_resource_update(self.resource, self.resource.stack.t.id, set(), 'engine-id', self.stack, None)
        self.assertFalse(mock_create.called)
        self.assertTrue(mock_update.called)

    @mock.patch.object(resource.Resource, 'create_convergence')
    @mock.patch.object(resource.Resource, 'update_convergence')
    def test_check_resource_update_update_action(self, mock_update, mock_create):
        self.resource.action = 'UPDATE'
        check_resource.check_resource_update(self.resource, self.resource.stack.t.id, set(), 'engine-id', self.stack, None)
        self.assertFalse(mock_create.called)
        self.assertTrue(mock_update.called)

    @mock.patch.object(resource.Resource, 'delete_convergence')
    def test_check_resource_cleanup_delete(self, mock_delete):
        self.resource.current_template_id = 'new-template-id'
        check_resource.check_resource_cleanup(self.resource, self.resource.stack.t.id, 'engine-id', self.stack.timeout_secs(), None)
        self.assertTrue(mock_delete.called)

    def test_check_message_raises_cancel_exception(self):
        msg_queue = eventlet.queue.LightQueue()
        msg_queue.put_nowait(rpc_api.THREAD_CANCEL)
        self.assertRaises(check_resource.CancelOperation, check_resource._check_for_message, msg_queue)