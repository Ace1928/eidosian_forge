import datetime
import http.client as http
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_serialization import jsonutils
import webob
import glance.api.v2.tasks
from glance.common import timeutils
import glance.domain
import glance.gateway
from glance.tests.unit import base
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
class TestTasksController(test_utils.BaseTestCase):

    def setUp(self):
        super(TestTasksController, self).setUp()
        self.db = unit_test_utils.FakeDB(initialize=False)
        self.policy = unit_test_utils.FakePolicyEnforcer()
        self.notifier = unit_test_utils.FakeNotifier()
        self.store = unit_test_utils.FakeStoreAPI()
        self._create_tasks()
        self.controller = glance.api.v2.tasks.TasksController(self.db, self.policy, self.notifier, self.store)
        self.gateway = glance.gateway.Gateway(self.db, self.store, self.notifier, self.policy)

    def _create_tasks(self):
        now = timeutils.utcnow()
        times = [now + datetime.timedelta(seconds=5 * i) for i in range(4)]
        self.tasks = [_db_fixture(UUID1, owner=TENANT1, created_at=times[0], updated_at=times[0]), _db_fixture(UUID2, owner=TENANT2, type='import', created_at=times[1], updated_at=times[1]), _db_fixture(UUID3, owner=TENANT3, type='import', created_at=times[2], updated_at=times[2]), _db_fixture(UUID4, owner=TENANT4, type='import', created_at=times[3], updated_at=times[3])]
        [self.db.task_create(None, task) for task in self.tasks]

    def test_index(self):
        self.config(limit_param_default=1, api_limit_max=3)
        request = unit_test_utils.get_fake_request()
        output = self.controller.index(request)
        self.assertEqual(1, len(output['tasks']))
        actual = set([task.task_id for task in output['tasks']])
        expected = set([UUID1])
        self.assertEqual(expected, actual)

    def test_index_admin(self):
        request = unit_test_utils.get_fake_request(is_admin=True)
        output = self.controller.index(request)
        self.assertEqual(4, len(output['tasks']))

    def test_index_return_parameters(self):
        self.config(limit_param_default=1, api_limit_max=4)
        request = unit_test_utils.get_fake_request(is_admin=True)
        output = self.controller.index(request, marker=UUID3, limit=1, sort_key='created_at', sort_dir='desc')
        self.assertEqual(1, len(output['tasks']))
        actual = set([task.task_id for task in output['tasks']])
        expected = set([UUID2])
        self.assertEqual(expected, actual)
        self.assertEqual(UUID2, output['next_marker'])

    def test_index_next_marker(self):
        self.config(limit_param_default=1, api_limit_max=3)
        request = unit_test_utils.get_fake_request(is_admin=True)
        output = self.controller.index(request, marker=UUID3, limit=2)
        self.assertEqual(2, len(output['tasks']))
        actual = set([task.task_id for task in output['tasks']])
        expected = set([UUID2, UUID1])
        self.assertEqual(expected, actual)
        self.assertEqual(UUID1, output['next_marker'])

    def test_index_no_next_marker(self):
        self.config(limit_param_default=1, api_limit_max=3)
        request = unit_test_utils.get_fake_request(is_admin=True)
        output = self.controller.index(request, marker=UUID1, limit=2)
        self.assertEqual(0, len(output['tasks']))
        actual = set([task.task_id for task in output['tasks']])
        expected = set([])
        self.assertEqual(expected, actual)
        self.assertNotIn('next_marker', output)

    def test_index_with_id_filter(self):
        request = unit_test_utils.get_fake_request('/tasks?id=%s' % UUID1)
        output = self.controller.index(request, filters={'id': UUID1})
        self.assertEqual(1, len(output['tasks']))
        actual = set([task.task_id for task in output['tasks']])
        expected = set([UUID1])
        self.assertEqual(expected, actual)

    def test_index_with_filters_return_many(self):
        path = '/tasks?status=pending'
        request = unit_test_utils.get_fake_request(path, is_admin=True)
        output = self.controller.index(request, filters={'status': 'pending'})
        self.assertEqual(4, len(output['tasks']))
        actual = set([task.task_id for task in output['tasks']])
        expected = set([UUID1, UUID2, UUID3, UUID4])
        self.assertEqual(sorted(expected), sorted(actual))

    def test_index_with_many_filters(self):
        url = '/tasks?status=pending&type=import'
        request = unit_test_utils.get_fake_request(url, is_admin=True)
        output = self.controller.index(request, filters={'status': 'pending', 'type': 'import', 'owner': TENANT1})
        self.assertEqual(1, len(output['tasks']))
        actual = set([task.task_id for task in output['tasks']])
        expected = set([UUID1])
        self.assertEqual(expected, actual)

    def test_index_with_marker(self):
        self.config(limit_param_default=1, api_limit_max=3)
        path = '/tasks'
        request = unit_test_utils.get_fake_request(path, is_admin=True)
        output = self.controller.index(request, marker=UUID3)
        actual = set([task.task_id for task in output['tasks']])
        self.assertEqual(1, len(actual))
        self.assertIn(UUID2, actual)

    def test_index_with_limit(self):
        path = '/tasks'
        limit = 2
        request = unit_test_utils.get_fake_request(path, is_admin=True)
        output = self.controller.index(request, limit=limit)
        actual = set([task.task_id for task in output['tasks']])
        self.assertEqual(limit, len(actual))

    def test_index_greater_than_limit_max(self):
        self.config(limit_param_default=1, api_limit_max=3)
        path = '/tasks'
        request = unit_test_utils.get_fake_request(path, is_admin=True)
        output = self.controller.index(request, limit=4)
        actual = set([task.task_id for task in output['tasks']])
        self.assertEqual(3, len(actual))
        self.assertNotIn(output['next_marker'], output)

    def test_index_default_limit(self):
        self.config(limit_param_default=1, api_limit_max=3)
        path = '/tasks'
        request = unit_test_utils.get_fake_request(path)
        output = self.controller.index(request)
        actual = set([task.task_id for task in output['tasks']])
        self.assertEqual(1, len(actual))

    def test_index_with_sort_dir(self):
        path = '/tasks'
        request = unit_test_utils.get_fake_request(path, is_admin=True)
        output = self.controller.index(request, sort_dir='asc', limit=3)
        actual = [task.task_id for task in output['tasks']]
        self.assertEqual(3, len(actual))
        self.assertEqual([UUID1, UUID2, UUID3], actual)

    def test_index_with_sort_key(self):
        path = '/tasks'
        request = unit_test_utils.get_fake_request(path, is_admin=True)
        output = self.controller.index(request, sort_key='created_at', limit=3)
        actual = [task.task_id for task in output['tasks']]
        self.assertEqual(3, len(actual))
        self.assertEqual(UUID4, actual[0])
        self.assertEqual(UUID3, actual[1])
        self.assertEqual(UUID2, actual[2])

    def test_index_with_marker_not_found(self):
        fake_uuid = str(uuid.uuid4())
        path = '/tasks'
        request = unit_test_utils.get_fake_request(path)
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.index, request, marker=fake_uuid)

    def test_index_with_marker_is_not_like_uuid(self):
        marker = 'INVALID_UUID'
        path = '/tasks'
        request = unit_test_utils.get_fake_request(path)
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.index, request, marker=marker)

    def test_index_invalid_sort_key(self):
        path = '/tasks'
        request = unit_test_utils.get_fake_request(path)
        self.assertRaises(webob.exc.HTTPBadRequest, self.controller.index, request, sort_key='foo')

    def test_index_zero_tasks(self):
        self.db.reset()
        request = unit_test_utils.get_fake_request()
        output = self.controller.index(request)
        self.assertEqual([], output['tasks'])

    def test_get(self):
        request = unit_test_utils.get_fake_request()
        task = self.controller.get(request, task_id=UUID1)
        self.assertEqual(UUID1, task.task_id)
        self.assertEqual('import', task.type)

    def test_get_non_existent(self):
        request = unit_test_utils.get_fake_request()
        task_id = str(uuid.uuid4())
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.get, request, task_id)

    def test_get_not_allowed(self):
        request = unit_test_utils.get_fake_request()
        self.assertEqual(TENANT1, request.context.project_id)
        self.assertRaises(webob.exc.HTTPNotFound, self.controller.get, request, UUID4)

    @mock.patch('glance.api.common.get_thread_pool')
    @mock.patch.object(glance.gateway.Gateway, 'get_task_factory')
    @mock.patch.object(glance.gateway.Gateway, 'get_task_executor_factory')
    @mock.patch.object(glance.gateway.Gateway, 'get_task_repo')
    def test_create(self, mock_get_task_repo, mock_get_task_executor_factory, mock_get_task_factory, mock_get_thread_pool):
        request = unit_test_utils.get_fake_request()
        task = {'type': 'import', 'input': {'import_from': 'swift://cloud.foo/myaccount/mycontainer/path', 'import_from_format': 'qcow2', 'image_properties': {}}}
        get_task_factory = mock.Mock()
        mock_get_task_factory.return_value = get_task_factory
        new_task = mock.Mock()
        get_task_factory.new_task.return_value = new_task
        new_task.run.return_value = mock.ANY
        get_task_executor_factory = mock.Mock()
        mock_get_task_executor_factory.return_value = get_task_executor_factory
        get_task_executor_factory.new_task_executor.return_value = mock.Mock()
        get_task_repo = mock.Mock()
        mock_get_task_repo.return_value = get_task_repo
        get_task_repo.add.return_value = mock.Mock()
        self.controller.create(request, task=task)
        self.assertEqual(1, get_task_factory.new_task.call_count)
        self.assertEqual(1, get_task_repo.add.call_count)
        self.assertEqual(1, get_task_executor_factory.new_task_executor.call_count)
        mock_get_thread_pool.assert_called_once_with('tasks_pool')
        mock_get_thread_pool.return_value.spawn.assert_called_once_with(new_task.run, get_task_executor_factory.new_task_executor.return_value)

    @mock.patch('glance.common.scripts.utils.get_image_data_iter')
    @mock.patch('glance.common.scripts.utils.validate_location_uri')
    def test_create_with_live_time(self, mock_validate_location_uri, mock_get_image_data_iter):
        self.skipTest('Something wrong, this test touches registry')
        request = unit_test_utils.get_fake_request()
        task = {'type': 'import', 'input': {'import_from': 'http://download.cirros-cloud.net/0.3.4/cirros-0.3.4-x86_64-disk.img', 'import_from_format': 'qcow2', 'image_properties': {'disk_format': 'qcow2', 'container_format': 'bare', 'name': 'test-task'}}}
        new_task = self.controller.create(request, task=task)
        executor_factory = self.gateway.get_task_executor_factory(request.context)
        task_executor = executor_factory.new_task_executor(request.context)
        task_executor.begin_processing(new_task.task_id)
        success_task = self.controller.get(request, new_task.task_id)
        task_live_time = success_task.expires_at.replace(second=0, microsecond=0) - success_task.updated_at.replace(second=0, microsecond=0)
        task_live_time_hour = task_live_time.days * 24 + task_live_time.seconds / 3600
        self.assertEqual(CONF.task.task_time_to_live, task_live_time_hour)

    def test_create_with_wrong_import_form(self):
        request = unit_test_utils.get_fake_request()
        wrong_import_from = ['swift://cloud.foo/myaccount/mycontainer/path', 'file:///path', 'cinder://volume-id']
        executor_factory = self.gateway.get_task_executor_factory(request.context)
        task_repo = self.gateway.get_task_repo(request.context)
        for import_from in wrong_import_from:
            task = {'type': 'import', 'input': {'import_from': import_from, 'import_from_format': 'qcow2', 'image_properties': {'disk_format': 'qcow2', 'container_format': 'bare', 'name': 'test-task'}}}
            new_task = self.controller.create(request, task=task)
            task_executor = executor_factory.new_task_executor(request.context)
            task_executor.begin_processing(new_task.task_id)
            final_task = task_repo.get(new_task.task_id)
            self.assertEqual('failure', final_task.status)
            if import_from.startswith('file:///'):
                msg = 'File based imports are not allowed. Please use a non-local source of image data.'
            else:
                supported = ['http']
                msg = 'The given uri is not valid. Please specify a valid uri from the following list of supported uri %(supported)s' % {'supported': supported}
            self.assertEqual(msg, final_task.message)

    def test_create_with_properties_missed(self):
        request = unit_test_utils.get_fake_request()
        executor_factory = self.gateway.get_task_executor_factory(request.context)
        task_repo = self.gateway.get_task_repo(request.context)
        task = {'type': 'import', 'input': {'import_from': 'swift://cloud.foo/myaccount/mycontainer/path', 'import_from_format': 'qcow2'}}
        new_task = self.controller.create(request, task=task)
        task_executor = executor_factory.new_task_executor(request.context)
        task_executor.begin_processing(new_task.task_id)
        final_task = task_repo.get(new_task.task_id)
        self.assertEqual('failure', final_task.status)
        msg = "Input does not contain 'image_properties' field"
        self.assertEqual(msg, final_task.message)

    @mock.patch.object(glance.gateway.Gateway, 'get_task_factory')
    def test_notifications_on_create(self, mock_get_task_factory):
        request = unit_test_utils.get_fake_request()
        new_task = mock.MagicMock(type='import')
        mock_get_task_factory.new_task.return_value = new_task
        new_task.run.return_value = mock.ANY
        task = {'type': 'import', 'input': {'import_from': 'http://cloud.foo/myaccount/mycontainer/path', 'import_from_format': 'qcow2', 'image_properties': {}}}
        task = self.controller.create(request, task=task)
        output_logs = [nlog for nlog in self.notifier.get_logs() if nlog['event_type'] == 'task.create']
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('task.create', output_log['event_type'])