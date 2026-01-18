from datetime import datetime
from datetime import timedelta
from unittest import mock
from oslo_config import cfg
from heat.common import template_format
from heat.engine import environment
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import snapshot as snapshot_objects
from heat.objects import stack as stack_object
from heat.objects import sync_point as sync_point_object
from heat.rpc import worker_client
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
@mock.patch.object(parser.Stack, '_persist_state')
class TestConvgStackStateSet(common.HeatTestCase):

    def setUp(self):
        super(TestConvgStackStateSet, self).setUp()
        cfg.CONF.set_override('convergence_engine', True)
        self.stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.wp_template, convergence=True)

    def test_state_set_create_adopt_update_delete_rollback_complete(self, mock_ps):
        mock_ps.return_value = 'updated'
        ret_val = self.stack.state_set(self.stack.CREATE, self.stack.COMPLETE, 'Create complete')
        self.assertTrue(mock_ps.called)
        self.assertEqual('updated', ret_val)
        mock_ps.reset_mock()
        ret_val = self.stack.state_set(self.stack.UPDATE, self.stack.COMPLETE, 'Update complete')
        self.assertTrue(mock_ps.called)
        self.assertEqual('updated', ret_val)
        mock_ps.reset_mock()
        ret_val = self.stack.state_set(self.stack.ROLLBACK, self.stack.COMPLETE, 'Rollback complete')
        self.assertTrue(mock_ps.called)
        self.assertEqual('updated', ret_val)
        mock_ps.reset_mock()
        ret_val = self.stack.state_set(self.stack.DELETE, self.stack.COMPLETE, 'Delete complete')
        self.assertTrue(mock_ps.called)
        self.assertEqual('updated', ret_val)
        mock_ps.reset_mock()
        ret_val = self.stack.state_set(self.stack.ADOPT, self.stack.COMPLETE, 'Adopt complete')
        self.assertTrue(mock_ps.called)
        self.assertEqual('updated', ret_val)

    def test_state_set_stack_suspend(self, mock_ps):
        mock_ps.return_value = 'updated'
        self.stack.state_set(self.stack.SUSPEND, self.stack.IN_PROGRESS, 'Suspend started')
        self.assertTrue(mock_ps.called)
        mock_ps.reset_mock()
        self.stack.state_set(self.stack.SUSPEND, self.stack.COMPLETE, 'Suspend complete')
        self.assertFalse(mock_ps.called)

    def test_state_set_stack_resume(self, mock_ps):
        self.stack.state_set(self.stack.RESUME, self.stack.IN_PROGRESS, 'Resume started')
        self.assertTrue(mock_ps.called)
        mock_ps.reset_mock()
        self.stack.state_set(self.stack.RESUME, self.stack.COMPLETE, 'Resume complete')
        self.assertFalse(mock_ps.called)

    def test_state_set_stack_snapshot(self, mock_ps):
        self.stack.state_set(self.stack.SNAPSHOT, self.stack.IN_PROGRESS, 'Snapshot started')
        self.assertTrue(mock_ps.called)
        mock_ps.reset_mock()
        self.stack.state_set(self.stack.SNAPSHOT, self.stack.COMPLETE, 'Snapshot complete')
        self.assertFalse(mock_ps.called)

    def test_state_set_stack_restore(self, mock_ps):
        mock_ps.return_value = 'updated'
        ret_val = self.stack.state_set(self.stack.RESTORE, self.stack.IN_PROGRESS, 'Restore started')
        self.assertTrue(mock_ps.called)
        self.assertEqual('updated', ret_val)
        mock_ps.reset_mock()
        ret_val = self.stack.state_set(self.stack.RESTORE, self.stack.COMPLETE, 'Restore complete')
        self.assertTrue(mock_ps.called)
        self.assertEqual('updated', ret_val)