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
@mock.patch.object(worker_client.WorkerClient, 'check_resource')
class StackConvergenceCreateUpdateDeleteTest(common.HeatTestCase):

    def setUp(self):
        super(StackConvergenceCreateUpdateDeleteTest, self).setUp()
        cfg.CONF.set_override('convergence_engine', True)
        self.stack = None

    @mock.patch.object(parser.Stack, 'mark_complete')
    def test_converge_empty_template(self, mock_mc, mock_cr):
        empty_tmpl = templatem.Template.create_empty_template()
        stack = parser.Stack(utils.dummy_context(), 'empty_tmpl_stack', empty_tmpl, convergence=True)
        stack.store()
        stack.thread_group_mgr = tools.DummyThreadGroupManager()
        stack.converge_stack(template=stack.t, action=stack.CREATE)
        self.assertFalse(mock_cr.called)
        mock_mc.assert_called_once_with()

    def test_conv_wordpress_single_instance_stack_create(self, mock_cr):
        stack = tools.get_stack('test_stack', utils.dummy_context(), convergence=True)
        stack.store()
        stack.converge_stack(template=stack.t, action=stack.CREATE)
        self.assertIsNone(stack.ext_rsrcs_db)
        self.assertEqual([((1, True), None)], list(stack.convergence_dependencies._graph.edges()))
        stack_db = stack_object.Stack.get_by_id(stack.context, stack.id)
        self.assertIsNotNone(stack_db.current_traversal)
        self.assertIsNotNone(stack_db.raw_template_id)
        self.assertIsNone(stack_db.prev_raw_template_id)
        self.assertTrue(stack_db.convergence)
        self.assertEqual({'edges': [[[1, True], None]]}, stack_db.current_deps)
        leaves = set(stack.convergence_dependencies.leaves())
        expected_calls = []
        for rsrc_id, is_update in sorted(leaves, key=lambda n: n.is_update):
            expected_calls.append(mock.call.worker_client.WorkerClient.check_resource(stack.context, rsrc_id, stack.current_traversal, {'input_data': {}}, is_update, None, False))
        self.assertEqual(expected_calls, mock_cr.mock_calls)

    def test_conv_string_five_instance_stack_create(self, mock_cr):
        stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
        stack.store()
        stack.converge_stack(template=stack.t, action=stack.CREATE)
        self.assertIsNone(stack.ext_rsrcs_db)
        self.assertEqual([((1, True), (3, True)), ((2, True), (3, True)), ((3, True), (4, True)), ((3, True), (5, True))], sorted(stack.convergence_dependencies._graph.edges()))
        stack_db = stack_object.Stack.get_by_id(stack.context, stack.id)
        self.assertIsNotNone(stack_db.current_traversal)
        self.assertIsNotNone(stack_db.raw_template_id)
        self.assertIsNone(stack_db.prev_raw_template_id)
        self.assertTrue(stack_db.convergence)
        self.assertEqual(sorted([[[3, True], [5, True]], [[3, True], [4, True]], [[1, True], [3, True]], [[2, True], [3, True]]]), sorted(stack_db.current_deps['edges']))
        for entity_id in [5, 4, 3, 2, 1, stack_db.id]:
            sync_point = sync_point_object.SyncPoint.get_by_key(stack_db._context, entity_id, stack_db.current_traversal, True)
            self.assertIsNotNone(sync_point)
            self.assertEqual(stack_db.id, sync_point.stack_id)
        leaves = set(stack.convergence_dependencies.leaves())
        expected_calls = []
        for rsrc_id, is_update in sorted(leaves, key=lambda n: n.is_update):
            expected_calls.append(mock.call.worker_client.WorkerClient.check_resource(stack.context, rsrc_id, stack.current_traversal, {'input_data': {}}, is_update, None, False))
        self.assertEqual(expected_calls, mock_cr.mock_calls)

    def _mock_convg_db_update_requires(self):
        """Updates requires column of resources.

        Required for testing the generation of convergence dependency graph
        on an update.
        """
        requires = dict()
        for rsrc_id, is_update in self.stack.convergence_dependencies:
            if is_update:
                reqs = self.stack.convergence_dependencies.requires((rsrc_id, is_update))
                requires[rsrc_id] = list({id for id, is_update in reqs})
        rsrcs_db = resource_objects.Resource.get_all_active_by_stack(self.stack.context, self.stack.id)
        for rsrc_id, rsrc in rsrcs_db.items():
            if rsrc.id in requires:
                rsrcs_db[rsrc_id].requires = requires[rsrc.id]
        return rsrcs_db

    def test_conv_string_five_instance_stack_update(self, mock_cr):
        stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
        stack.store()
        stack.converge_stack(template=stack.t, action=stack.CREATE)
        curr_stack_db = stack_object.Stack.get_by_id(stack.context, stack.id)
        curr_stack = parser.Stack.load(curr_stack_db._context, stack=curr_stack_db)
        t2 = template_format.parse(tools.string_template_five_update)
        template2 = templatem.Template(t2, env=environment.Environment({'KeyName2': 'test2'}))
        self.stack = stack
        with mock.patch.object(parser.Stack, 'db_active_resources_get', side_effect=self._mock_convg_db_update_requires):
            curr_stack.thread_group_mgr = tools.DummyThreadGroupManager()
            curr_stack.converge_stack(template=template2, action=stack.UPDATE)
        self.assertIsNotNone(curr_stack.ext_rsrcs_db)
        deps = curr_stack.convergence_dependencies
        self.assertEqual([((3, False), (1, False)), ((3, False), (2, False)), ((4, False), (3, False)), ((4, False), (4, True)), ((5, False), (3, False)), ((5, False), (5, True)), ((6, True), (8, True)), ((7, True), (8, True)), ((8, True), (4, True)), ((8, True), (5, True))], sorted(deps._graph.edges()))
        stack_db = stack_object.Stack.get_by_id(curr_stack.context, curr_stack.id)
        self.assertIsNotNone(stack_db.raw_template_id)
        self.assertIsNotNone(stack_db.current_traversal)
        self.assertIsNotNone(stack_db.prev_raw_template_id)
        self.assertTrue(stack_db.convergence)
        self.assertEqual(sorted([[[7, True], [8, True]], [[8, True], [5, True]], [[8, True], [4, True]], [[6, True], [8, True]], [[3, False], [2, False]], [[3, False], [1, False]], [[5, False], [3, False]], [[5, False], [5, True]], [[4, False], [3, False]], [[4, False], [4, True]]]), sorted(stack_db.current_deps['edges']))
        '\n        To visualize:\n\n        G(7, True)       H(6, True)\n            \\                 /\n              \\             /           B(4, False)   A(5, False)\n                \\         /               /       \\  /    /\n                  \\     /            /           /\n               F(8, True)       /             /     \\  /\n                    /  \\    /             /     C(3, False)\n                  /      \\            /            /    \\\n                /     /    \\      /\n              /    /         \\ /                /          \\\n        B(4, True)      A(5, True)       D(2, False)    E(1, False)\n\n        Leaves are at the bottom\n        '
        for entity_id in [8, 7, 6, 5, 4, stack_db.id]:
            sync_point = sync_point_object.SyncPoint.get_by_key(stack_db._context, entity_id, stack_db.current_traversal, True)
            self.assertIsNotNone(sync_point)
            self.assertEqual(stack_db.id, sync_point.stack_id)
        for entity_id in [5, 4, 3, 2, 1]:
            sync_point = sync_point_object.SyncPoint.get_by_key(stack_db._context, entity_id, stack_db.current_traversal, False)
            self.assertIsNotNone(sync_point)
            self.assertEqual(stack_db.id, sync_point.stack_id)
        leaves = set(stack.convergence_dependencies.leaves())
        expected_calls = []
        for rsrc_id, is_update in sorted(leaves, key=lambda n: n.is_update):
            expected_calls.append(mock.call.worker_client.WorkerClient.check_resource(stack.context, rsrc_id, stack.current_traversal, {'input_data': {}}, is_update, None, False))
        leaves = set(curr_stack.convergence_dependencies.leaves())
        for rsrc_id, is_update in sorted(leaves, key=lambda n: n.is_update):
            expected_calls.append(mock.call.worker_client.WorkerClient.check_resource(curr_stack.context, rsrc_id, curr_stack.current_traversal, {'input_data': {}}, is_update, None, False))
        self.assertEqual(expected_calls, mock_cr.mock_calls)

    def test_conv_empty_template_stack_update_delete(self, mock_cr):
        stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
        stack.store()
        stack.converge_stack(template=stack.t, action=stack.CREATE)
        template2 = templatem.Template.create_empty_template(version=stack.t.version)
        curr_stack_db = stack_object.Stack.get_by_id(stack.context, stack.id)
        curr_stack = parser.Stack.load(curr_stack_db._context, stack=curr_stack_db)
        self.stack = stack
        with mock.patch.object(parser.Stack, 'db_active_resources_get', side_effect=self._mock_convg_db_update_requires):
            curr_stack.thread_group_mgr = tools.DummyThreadGroupManager()
            curr_stack.converge_stack(template=template2, action=stack.DELETE)
        self.assertIsNotNone(curr_stack.ext_rsrcs_db)
        deps = curr_stack.convergence_dependencies
        self.assertEqual([((3, False), (1, False)), ((3, False), (2, False)), ((4, False), (3, False)), ((5, False), (3, False))], sorted(deps._graph.edges()))
        stack_db = stack_object.Stack.get_by_id(curr_stack.context, curr_stack.id)
        self.assertIsNotNone(stack_db.current_traversal)
        self.assertIsNotNone(stack_db.prev_raw_template_id)
        self.assertEqual(sorted([[[3, False], [2, False]], [[3, False], [1, False]], [[5, False], [3, False]], [[4, False], [3, False]]]), sorted(stack_db.current_deps['edges']))
        for entity_id in [5, 4, 3, 2, 1, stack_db.id]:
            is_update = False
            if entity_id == stack_db.id:
                is_update = True
            sync_point = sync_point_object.SyncPoint.get_by_key(stack_db._context, entity_id, stack_db.current_traversal, is_update)
            self.assertIsNotNone(sync_point, 'entity %s' % entity_id)
            self.assertEqual(stack_db.id, sync_point.stack_id)
        leaves = set(stack.convergence_dependencies.leaves())
        expected_calls = []
        for rsrc_id, is_update in sorted(leaves, key=lambda n: n.is_update):
            expected_calls.append(mock.call.worker_client.WorkerClient.check_resource(stack.context, rsrc_id, stack.current_traversal, {'input_data': {}}, is_update, None, False))
        leaves = set(curr_stack.convergence_dependencies.leaves())
        for rsrc_id, is_update in sorted(leaves, key=lambda n: n.is_update):
            expected_calls.append(mock.call.worker_client.WorkerClient.check_resource(curr_stack.context, rsrc_id, curr_stack.current_traversal, {'input_data': {}}, is_update, None, False))
        self.assertEqual(expected_calls, mock_cr.mock_calls)

    def test_mark_complete_purges_db(self, mock_cr):
        stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
        stack.store()
        stack.purge_db = mock.Mock()
        stack.mark_complete()
        self.assertTrue(stack.purge_db.called)

    def test_state_set_sets_empty_curr_trvsl_for_failed_stack(self, mock_cr):
        stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
        stack.status = stack.FAILED
        stack.store()
        stack.purge_db()
        self.assertEqual('', stack.current_traversal)

    @mock.patch.object(raw_template_object.RawTemplate, 'delete')
    def test_purge_db_deletes_previous_template(self, mock_tmpl_delete, mock_cr):
        stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
        stack.prev_raw_template_id = 10
        stack.purge_db()
        self.assertTrue(mock_tmpl_delete.called)

    @mock.patch.object(parser.Stack, '_delete_credentials')
    @mock.patch.object(stack_object.Stack, 'delete')
    def test_purge_db_deletes_creds(self, mock_delete_stack, mock_creds_delete, mock_cr):
        stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
        reason = 'stack delete complete'
        mock_creds_delete.return_value = (stack.COMPLETE, reason)
        stack.state_set(stack.DELETE, stack.COMPLETE, reason)
        stack.purge_db()
        self.assertTrue(mock_creds_delete.called)
        self.assertTrue(mock_delete_stack.called)

    @mock.patch.object(parser.Stack, '_delete_credentials')
    @mock.patch.object(stack_object.Stack, 'delete')
    def test_purge_db_deletes_creds_failed(self, mock_delete_stack, mock_creds_delete, mock_cr):
        stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
        reason = 'stack delete complete'
        failed_reason = 'Error deleting trust'
        mock_creds_delete.return_value = (stack.FAILED, failed_reason)
        stack.state_set(stack.DELETE, stack.COMPLETE, reason)
        stack.purge_db()
        self.assertTrue(mock_creds_delete.called)
        self.assertFalse(mock_delete_stack.called)
        self.assertEqual((stack.DELETE, stack.FAILED), stack.state)

    @mock.patch.object(raw_template_object.RawTemplate, 'delete')
    def test_purge_db_does_not_delete_previous_template_when_stack_fails(self, mock_tmpl_delete, mock_cr):
        stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
        stack.status = stack.FAILED
        stack.purge_db()
        self.assertFalse(mock_tmpl_delete.called)

    def test_purge_db_deletes_sync_points(self, mock_cr):
        stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
        stack.store()
        stack.purge_db()
        rows = sync_point_object.SyncPoint.delete_all_by_stack_and_traversal(stack.context, stack.id, stack.current_traversal)
        self.assertEqual(0, rows)

    @mock.patch.object(stack_object.Stack, 'delete')
    def test_purge_db_deletes_stack_for_deleted_stack(self, mock_stack_delete, mock_cr):
        stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
        stack.store()
        stack.state_set(stack.DELETE, stack.COMPLETE, 'test reason')
        stack.purge_db()
        self.assertTrue(mock_stack_delete.called)

    @mock.patch.object(resource_objects.Resource, 'purge_deleted')
    def test_purge_db_calls_rsrc_purge_deleted(self, mock_rsrc_purge_delete, mock_cr):
        stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
        stack.store()
        stack.purge_db()
        self.assertTrue(mock_rsrc_purge_delete.called)

    def test_get_best_existing_db_resource(self, mock_cr):
        stack = tools.get_stack('test_stack', utils.dummy_context(), template=tools.string_template_five, convergence=True)
        stack.prev_raw_template_id = 2
        stack.t.id = 3

        def db_resource(current_template_id, created_at=None, updated_at=None):
            db_res = resource_objects.Resource(stack.context)
            db_res['id'] = current_template_id
            db_res['name'] = 'A'
            db_res['current_template_id'] = current_template_id
            db_res['action'] = 'UPDATE' if updated_at else 'CREATE'
            db_res['status'] = 'COMPLETE'
            db_res['updated_at'] = updated_at
            db_res['created_at'] = created_at
            db_res['replaced_by'] = None
            return db_res
        start_time = datetime.utcfromtimestamp(0)

        def t(minutes):
            return start_time + timedelta(minutes=minutes)
        a_res_2 = db_resource(2)
        a_res_3 = db_resource(3)
        a_res_0 = db_resource(0, created_at=t(0), updated_at=t(1))
        a_res_1 = db_resource(1, created_at=t(2))
        existing_res = {a_res_2.id: a_res_2, a_res_3.id: a_res_3, a_res_0.id: a_res_0, a_res_1.id: a_res_1}
        stack.ext_rsrcs_db = existing_res
        best_res = stack._get_best_existing_rsrc_db('A')
        self.assertEqual(a_res_3.id, best_res.id)
        del existing_res[3]
        best_res = stack._get_best_existing_rsrc_db('A')
        self.assertEqual(a_res_2.id, best_res.id)
        del existing_res[2]
        best_res = stack._get_best_existing_rsrc_db('A')
        self.assertEqual(a_res_1.id, best_res.id)
        del existing_res[1]
        best_res = stack._get_best_existing_rsrc_db('A')
        self.assertEqual(a_res_0.id, best_res.id)

    @mock.patch.object(parser.Stack, '_converge_create_or_update')
    def test_updated_time_stack_create(self, mock_ccu, mock_cr):
        stack = parser.Stack(utils.dummy_context(), 'convg_updated_time_test', templatem.Template.create_empty_template(), convergence=True)
        stack.thread_group_mgr = tools.DummyThreadGroupManager()
        stack.converge_stack(template=stack.t, action=stack.CREATE)
        self.assertIsNone(stack.updated_time)
        self.assertTrue(mock_ccu.called)

    @mock.patch.object(parser.Stack, '_converge_create_or_update')
    def test_updated_time_stack_update(self, mock_ccu, mock_cr):
        tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'R1': {'Type': 'GenericResourceType'}}}
        stack = parser.Stack(utils.dummy_context(), 'updated_time_test', templatem.Template(tmpl), convergence=True)
        stack.thread_group_mgr = tools.DummyThreadGroupManager()
        stack.converge_stack(template=stack.t, action=stack.UPDATE)
        self.assertIsNotNone(stack.updated_time)
        self.assertTrue(mock_ccu.called)

    @mock.patch.object(parser.Stack, '_converge_create_or_update')
    @mock.patch.object(sync_point_object.SyncPoint, 'delete_all_by_stack_and_traversal')
    def test_sync_point_delete_stack_create(self, mock_syncpoint_del, mock_ccu, mock_cr):
        stack = parser.Stack(utils.dummy_context(), 'convg_updated_time_test', templatem.Template.create_empty_template(), convergence=True)
        stack.thread_group_mgr = tools.DummyThreadGroupManager()
        stack.converge_stack(template=stack.t, action=stack.CREATE)
        self.assertFalse(mock_syncpoint_del.called)
        self.assertTrue(mock_ccu.called)

    @mock.patch.object(parser.Stack, '_converge_create_or_update')
    @mock.patch.object(sync_point_object.SyncPoint, 'delete_all_by_stack_and_traversal')
    def test_sync_point_delete_stack_update(self, mock_syncpoint_del, mock_ccu, mock_cr):
        tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'R1': {'Type': 'GenericResourceType'}}}
        stack = parser.Stack(utils.dummy_context(), 'updated_time_test', templatem.Template(tmpl), convergence=True)
        stack.thread_group_mgr = tools.DummyThreadGroupManager()
        stack.current_traversal = 'prev_traversal'
        stack.converge_stack(template=stack.t, action=stack.UPDATE)
        self.assertTrue(mock_syncpoint_del.called)
        self.assertTrue(mock_ccu.called)

    def test_snapshot_delete(self, mock_cr):
        tmpl = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'R1': {'Type': 'GenericResourceType'}}}
        stack = parser.Stack(utils.dummy_context(), 'updated_time_test', templatem.Template(tmpl), convergence=True)
        stack.current_traversal = 'prev_traversal'
        stack.action, stack.status = (stack.CREATE, stack.COMPLETE)
        stack.store()
        stack.thread_group_mgr = tools.DummyThreadGroupManager()
        snapshot_values = {'stack_id': stack.id, 'name': 'fake_snapshot', 'tenant': stack.context.tenant_id, 'status': 'COMPLETE', 'data': None}
        snapshot_objects.Snapshot.create(stack.context, snapshot_values)
        stack.converge_stack(template=stack.t, action=stack.UPDATE)
        db_snapshot_obj = snapshot_objects.Snapshot.get_all_by_stack(stack.context, stack.id)
        self.assertEqual('fake_snapshot', db_snapshot_obj[0].name)
        self.assertEqual(stack.id, db_snapshot_obj[0].stack_id)
        stack.converge_stack(template=stack.t, action=stack.DELETE)
        self.assertEqual([], snapshot_objects.Snapshot.get_all_by_stack(stack.context, stack.id))
        self.assertTrue(mock_cr.called)