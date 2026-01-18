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
@mock.patch.object(check_resource, 'check_stack_complete')
@mock.patch.object(check_resource, 'propagate_check_resource')
@mock.patch.object(check_resource, 'check_resource_cleanup')
@mock.patch.object(check_resource, 'check_resource_update')
class CheckWorkflowCleanupTest(common.HeatTestCase):

    @mock.patch.object(worker_client.WorkerClient, 'check_resource', lambda *_: None)
    def setUp(self):
        super(CheckWorkflowCleanupTest, self).setUp()
        thread_group_mgr = mock.Mock()
        self.worker = worker.WorkerService('host-1', 'topic-1', 'engine_id', thread_group_mgr)
        self.worker._rpc_client = worker_client.WorkerClient()
        self.ctx = utils.dummy_context()
        tstack = tools.get_stack('check_workflow_create_stack', self.ctx, template=tools.string_template_five, convergence=True)
        tstack.converge_stack(tstack.t, action=tstack.CREATE)
        self.stack = stack.Stack.load(self.ctx, stack_id=tstack.id)
        self.stack.thread_group_mgr = tools.DummyThreadGroupManager()
        self.stack.converge_stack(self.stack.t, action=self.stack.DELETE)
        self.resource = self.stack['A']
        self.is_update = False
        self.graph_key = (self.resource.id, self.is_update)

    @mock.patch.object(resource.Resource, 'load')
    @mock.patch.object(stack.Stack, 'time_remaining')
    def test_is_cleanup_traversal(self, tr, mock_load, mock_cru, mock_crc, mock_pcr, mock_csc):
        tr.return_value = 317
        mock_load.return_value = (self.resource, self.stack, self.stack)
        self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
        self.assertFalse(mock_cru.called)
        mock_crc.assert_called_once_with(self.resource, self.resource.stack.t.id, self.worker.engine_id, tr(), mock.ANY)

    @mock.patch.object(stack.Stack, 'time_remaining')
    def test_is_cleanup_traversal_raise_update_inprogress(self, tr, mock_cru, mock_crc, mock_pcr, mock_csc):
        mock_crc.side_effect = exception.UpdateInProgress
        tr.return_value = 317
        self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, None)
        mock_crc.assert_called_once_with(self.resource, self.resource.stack.t.id, self.worker.engine_id, tr(), mock.ANY)
        self.assertFalse(mock_cru.called)
        self.assertFalse(mock_pcr.called)
        self.assertFalse(mock_csc.called)

    def test_check_resource_does_not_propagate_on_cancelling_cleanup(self, mock_cru, mock_crc, mock_pcr, mock_csc):
        mock_crc.side_effect = check_resource.CancelOperation
        self.worker.check_resource(self.ctx, self.resource.id, self.stack.current_traversal, {}, self.is_update, {})
        self.assertFalse(mock_pcr.called)
        self.assertFalse(mock_csc.called)