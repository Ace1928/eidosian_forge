import datetime
from unittest import mock
import glance_store
from oslo_config import cfg
import oslo_messaging
import webob
import glance.async_
from glance.common import exception
from glance.common import timeutils
import glance.context
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
from glance.tests import utils
class TestTaskNotifications(utils.BaseTestCase):
    """Test Task Notifications work"""

    def setUp(self):
        super(TestTaskNotifications, self).setUp()
        task_input = {'loc': 'fake'}
        self.task_stub = TaskStub(task_id='aaa', task_type='import', status='pending', owner=TENANT2, expires_at=None, created_at=DATETIME, updated_at=DATETIME, image_id='fake_image_id', user_id='fake_user', request_id='fake_request_id')
        self.task = Task(task_id='aaa', task_type='import', status='pending', owner=TENANT2, expires_at=None, created_at=DATETIME, updated_at=DATETIME, task_input=task_input, result='res', message='blah', image_id='fake_image_id', user_id='fake_user', request_id='fake_request_id')
        self.context = glance.context.RequestContext(tenant=TENANT2, user=USER1)
        self.task_repo_stub = TaskRepoStub()
        self.notifier = unit_test_utils.FakeNotifier()
        self.task_repo_proxy = glance.notifier.TaskRepoProxy(self.task_repo_stub, self.context, self.notifier)
        self.task_proxy = glance.notifier.TaskProxy(self.task, self.context, self.notifier)
        self.task_stub_proxy = glance.notifier.TaskStubProxy(self.task_stub, self.context, self.notifier)
        self.patcher = mock.patch.object(timeutils, 'utcnow')
        mock_utcnow = self.patcher.start()
        mock_utcnow.return_value = datetime.datetime.utcnow()

    def tearDown(self):
        super(TestTaskNotifications, self).tearDown()
        self.patcher.stop()

    def test_task_create_notification(self):
        self.task_repo_proxy.add(self.task_stub_proxy)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('task.create', output_log['event_type'])
        self.assertEqual(self.task.task_id, output_log['payload']['id'])
        self.assertEqual(timeutils.isotime(self.task.updated_at), output_log['payload']['updated_at'])
        self.assertEqual(timeutils.isotime(self.task.created_at), output_log['payload']['created_at'])
        if 'location' in output_log['payload']:
            self.fail('Notification contained location field.')
        self.assertNotIn('image_id', output_log['payload'])
        self.assertNotIn('user_id', output_log['payload'])
        self.assertNotIn('request_id', output_log['payload'])

    def test_task_create_notification_disabled(self):
        self.config(disabled_notifications=['task.create'])
        self.task_repo_proxy.add(self.task_stub_proxy)
        output_logs = self.notifier.get_logs()
        self.assertEqual(0, len(output_logs))

    def test_task_delete_notification(self):
        now = timeutils.isotime()
        self.task_repo_proxy.remove(self.task_stub_proxy)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('task.delete', output_log['event_type'])
        self.assertEqual(self.task.task_id, output_log['payload']['id'])
        self.assertEqual(timeutils.isotime(self.task.updated_at), output_log['payload']['updated_at'])
        self.assertEqual(timeutils.isotime(self.task.created_at), output_log['payload']['created_at'])
        self.assertEqual(now, output_log['payload']['deleted_at'])
        if 'location' in output_log['payload']:
            self.fail('Notification contained location field.')
        self.assertNotIn('image_id', output_log['payload'])
        self.assertNotIn('user_id', output_log['payload'])
        self.assertNotIn('request_id', output_log['payload'])

    def test_task_delete_notification_disabled(self):
        self.config(disabled_notifications=['task.delete'])
        self.task_repo_proxy.remove(self.task_stub_proxy)
        output_logs = self.notifier.get_logs()
        self.assertEqual(0, len(output_logs))

    def test_task_run_notification(self):
        with mock.patch('glance.async_.TaskExecutor') as mock_executor:
            executor = mock_executor.return_value
            executor._run.return_value = mock.Mock()
            self.task_proxy.run(executor=mock_executor)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('task.run', output_log['event_type'])
        self.assertEqual(self.task.task_id, output_log['payload']['id'])
        self.assertNotIn(self.task.image_id, output_log['payload'])
        self.assertNotIn(self.task.user_id, output_log['payload'])
        self.assertNotIn(self.task.request_id, output_log['payload'])

    def test_task_run_notification_disabled(self):
        self.config(disabled_notifications=['task.run'])
        with mock.patch('glance.async_.TaskExecutor') as mock_executor:
            executor = mock_executor.return_value
            executor._run.return_value = mock.Mock()
            self.task_proxy.run(executor=mock_executor)
        output_logs = self.notifier.get_logs()
        self.assertEqual(0, len(output_logs))

    def test_task_processing_notification(self):
        self.task_proxy.begin_processing()
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('task.processing', output_log['event_type'])
        self.assertEqual(self.task.task_id, output_log['payload']['id'])
        self.assertNotIn('image_id', output_log['payload'])
        self.assertNotIn('user_id', output_log['payload'])
        self.assertNotIn('request_id', output_log['payload'])

    def test_task_processing_notification_disabled(self):
        self.config(disabled_notifications=['task.processing'])
        self.task_proxy.begin_processing()
        output_logs = self.notifier.get_logs()
        self.assertEqual(0, len(output_logs))

    def test_task_success_notification(self):
        self.task_proxy.begin_processing()
        self.task_proxy.succeed(result=None)
        output_logs = self.notifier.get_logs()
        self.assertEqual(2, len(output_logs))
        output_log = output_logs[1]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('task.success', output_log['event_type'])
        self.assertEqual(self.task.task_id, output_log['payload']['id'])
        self.assertNotIn('image_id', output_log['payload'])
        self.assertNotIn('user_id', output_log['payload'])
        self.assertNotIn('request_id', output_log['payload'])

    def test_task_success_notification_disabled(self):
        self.config(disabled_notifications=['task.processing', 'task.success'])
        self.task_proxy.begin_processing()
        self.task_proxy.succeed(result=None)
        output_logs = self.notifier.get_logs()
        self.assertEqual(0, len(output_logs))

    def test_task_failure_notification(self):
        self.task_proxy.fail(message=None)
        output_logs = self.notifier.get_logs()
        self.assertEqual(1, len(output_logs))
        output_log = output_logs[0]
        self.assertEqual('INFO', output_log['notification_type'])
        self.assertEqual('task.failure', output_log['event_type'])
        self.assertEqual(self.task.task_id, output_log['payload']['id'])
        self.assertNotIn('image_id', output_log['payload'])
        self.assertNotIn('user_id', output_log['payload'])
        self.assertNotIn('request_id', output_log['payload'])

    def test_task_failure_notification_disabled(self):
        self.config(disabled_notifications=['task.failure'])
        self.task_proxy.fail(message=None)
        output_logs = self.notifier.get_logs()
        self.assertEqual(0, len(output_logs))