import sys
from unittest import mock
import urllib.error
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_utils import units
import taskflow
import glance.async_.flows.api_image_import as import_flow
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance import context
from glance.domain import ExtraProperties
from glance import gateway
import glance.tests.utils as test_utils
from cursive import exception as cursive_exception
class TestImportToStoreTask(test_utils.BaseTestCase):

    def setUp(self):
        super(TestImportToStoreTask, self).setUp()
        self.gateway = gateway.Gateway()
        self.context = context.RequestContext(user_id=TENANT1, project_id=TENANT1, overwrite=False)
        self.img_factory = self.gateway.get_image_factory(self.context)

    def test_execute(self):
        wrapper = mock.MagicMock()
        action = mock.MagicMock()
        task_repo = mock.MagicMock()
        wrapper.__enter__.return_value = action
        image_import = import_flow._ImportToStore(TASK_ID1, TASK_TYPE, task_repo, wrapper, 'http://url', 'store1', False, True)
        with mock.patch.object(image_import, '_execute') as mock_execute:
            image_import.execute(mock.sentinel.path)
            mock_execute.assert_called_once_with(action, mock.sentinel.path)
        with mock.patch.object(image_import, '_execute') as mock_execute:
            image_import.execute()
            mock_execute.assert_called_once_with(action, None)

    def test_execute_body_with_store(self):
        image = mock.MagicMock()
        img_repo = mock.MagicMock()
        img_repo.get.return_value = image
        task_repo = mock.MagicMock()
        wrapper = import_flow.ImportActionWrapper(img_repo, IMAGE_ID1, TASK_ID1)
        image_import = import_flow._ImportToStore(TASK_ID1, TASK_TYPE, task_repo, wrapper, 'http://url', 'store1', False, True)
        action = mock.MagicMock()
        image_import._execute(action, mock.sentinel.path)
        action.set_image_data.assert_called_once_with(mock.sentinel.path, TASK_ID1, backend='store1', set_active=True, callback=image_import._status_callback)
        action.remove_importing_stores(['store1'])

    def test_execute_body_with_store_no_path(self):
        image = mock.MagicMock()
        img_repo = mock.MagicMock()
        img_repo.get.return_value = image
        task_repo = mock.MagicMock()
        wrapper = import_flow.ImportActionWrapper(img_repo, IMAGE_ID1, TASK_ID1)
        image_import = import_flow._ImportToStore(TASK_ID1, TASK_TYPE, task_repo, wrapper, 'http://url', 'store1', False, True)
        action = mock.MagicMock()
        image_import._execute(action, None)
        action.set_image_data.assert_called_once_with('http://url', TASK_ID1, backend='store1', set_active=True, callback=image_import._status_callback)
        action.remove_importing_stores(['store1'])

    def test_execute_body_without_store(self):
        image = mock.MagicMock()
        img_repo = mock.MagicMock()
        img_repo.get.return_value = image
        task_repo = mock.MagicMock()
        wrapper = import_flow.ImportActionWrapper(img_repo, IMAGE_ID1, TASK_ID1)
        image_import = import_flow._ImportToStore(TASK_ID1, TASK_TYPE, task_repo, wrapper, 'http://url', None, False, True)
        action = mock.MagicMock()
        image_import._execute(action, mock.sentinel.path)
        action.set_image_data.assert_called_once_with(mock.sentinel.path, TASK_ID1, backend=None, set_active=True, callback=image_import._status_callback)
        action.remove_importing_stores.assert_not_called()

    @mock.patch('glance.async_.flows.api_image_import.LOG.debug')
    @mock.patch('oslo_utils.timeutils.now')
    def test_status_callback_limits_rate(self, mock_now, mock_log):
        img_repo = mock.MagicMock()
        task_repo = mock.MagicMock()
        task_repo.get.return_value.status = 'processing'
        wrapper = import_flow.ImportActionWrapper(img_repo, IMAGE_ID1, TASK_ID1)
        image_import = import_flow._ImportToStore(TASK_ID1, TASK_TYPE, task_repo, wrapper, 'http://url', None, False, True)
        expected_calls = []
        log_call = mock.call('Image import %(image_id)s copied %(copied)i MiB', {'image_id': IMAGE_ID1, 'copied': 0})
        action = mock.MagicMock(image_id=IMAGE_ID1)
        mock_now.return_value = 1000
        image_import._status_callback(action, 32, 32)
        expected_calls.append(log_call)
        mock_log.assert_has_calls(expected_calls)
        image_import._status_callback(action, 32, 64)
        mock_log.assert_has_calls(expected_calls)
        mock_now.return_value += 190
        image_import._status_callback(action, 32, 96)
        mock_log.assert_has_calls(expected_calls)
        mock_now.return_value += 300
        image_import._status_callback(action, 32, 128)
        expected_calls.append(log_call)
        mock_log.assert_has_calls(expected_calls)
        mock_now.return_value += 150
        image_import._status_callback(action, 32, 128)
        mock_log.assert_has_calls(expected_calls)
        mock_now.return_value += 3600
        image_import._status_callback(action, 32, 128)
        expected_calls.append(log_call)
        mock_log.assert_has_calls(expected_calls)

    def test_raises_when_image_deleted(self):
        img_repo = mock.MagicMock()
        task_repo = mock.MagicMock()
        wrapper = import_flow.ImportActionWrapper(img_repo, IMAGE_ID1, TASK_ID1)
        image_import = import_flow._ImportToStore(TASK_ID1, TASK_TYPE, task_repo, wrapper, 'http://url', 'store1', False, True)
        image = self.img_factory.new_image(image_id=UUID1)
        image.status = 'deleted'
        img_repo.get.return_value = image
        self.assertRaises(exception.ImportTaskError, image_import.execute)

    @mock.patch('glance.async_.flows.api_image_import.image_import')
    def test_remove_store_from_property(self, mock_import):
        img_repo = mock.MagicMock()
        task_repo = mock.MagicMock()
        wrapper = import_flow.ImportActionWrapper(img_repo, IMAGE_ID1, TASK_ID1)
        image_import = import_flow._ImportToStore(TASK_ID1, TASK_TYPE, task_repo, wrapper, 'http://url', 'store1', True, True)
        extra_properties = {'os_glance_importing_to_stores': 'store1,store2', 'os_glance_import_task': TASK_ID1}
        image = self.img_factory.new_image(image_id=UUID1, extra_properties=extra_properties)
        img_repo.get.return_value = image
        image_import.execute()
        self.assertEqual(image.extra_properties['os_glance_importing_to_stores'], 'store2')

    def test_revert_updates_status_keys(self):
        img_repo = mock.MagicMock()
        task_repo = mock.MagicMock()
        wrapper = import_flow.ImportActionWrapper(img_repo, IMAGE_ID1, TASK_ID1)
        image_import = import_flow._ImportToStore(TASK_ID1, TASK_TYPE, task_repo, wrapper, 'http://url', 'store1', True, True)
        extra_properties = {'os_glance_importing_to_stores': 'store1,store2', 'os_glance_import_task': TASK_ID1}
        image = self.img_factory.new_image(image_id=UUID1, extra_properties=extra_properties)
        img_repo.get.return_value = image
        fail_key = 'os_glance_failed_import'
        pend_key = 'os_glance_importing_to_stores'
        image_import.revert(None)
        self.assertEqual('store2', image.extra_properties[pend_key])
        try:
            raise Exception('foo')
        except Exception:
            fake_exc_info = sys.exc_info()
        extra_properties = {'os_glance_importing_to_stores': 'store1,store2'}
        image_import.revert(taskflow.types.failure.Failure(fake_exc_info))
        self.assertEqual('store2', image.extra_properties[pend_key])
        self.assertEqual('store1', image.extra_properties[fail_key])

    @mock.patch('glance.async_.flows.api_image_import.image_import')
    def test_raises_when_all_stores_must_succeed(self, mock_import):
        img_repo = mock.MagicMock()
        task_repo = mock.MagicMock()
        wrapper = import_flow.ImportActionWrapper(img_repo, IMAGE_ID1, TASK_ID1)
        image_import = import_flow._ImportToStore(TASK_ID1, TASK_TYPE, task_repo, wrapper, 'http://url', 'store1', True, True)
        extra_properties = {'os_glance_import_task': TASK_ID1}
        image = self.img_factory.new_image(image_id=UUID1, extra_properties=extra_properties)
        img_repo.get.return_value = image
        mock_import.set_image_data.side_effect = cursive_exception.SignatureVerificationError('Signature verification failed')
        self.assertRaises(cursive_exception.SignatureVerificationError, image_import.execute)

    @mock.patch('glance.async_.flows.api_image_import.image_import')
    def test_doesnt_raise_when_not_all_stores_must_succeed(self, mock_import):
        img_repo = mock.MagicMock()
        task_repo = mock.MagicMock()
        wrapper = import_flow.ImportActionWrapper(img_repo, IMAGE_ID1, TASK_ID1)
        image_import = import_flow._ImportToStore(TASK_ID1, TASK_TYPE, task_repo, wrapper, 'http://url', 'store1', False, True)
        extra_properties = {'os_glance_import_task': TASK_ID1}
        image = self.img_factory.new_image(image_id=UUID1, extra_properties=extra_properties)
        img_repo.get.return_value = image
        mock_import.set_image_data.side_effect = cursive_exception.SignatureVerificationError('Signature verification failed')
        try:
            image_import.execute()
            self.assertEqual(image.extra_properties['os_glance_failed_import'], 'store1')
        except cursive_exception.SignatureVerificationError:
            self.fail("Exception shouldn't be raised")

    @mock.patch('glance.common.scripts.utils.get_task')
    def test_status_callback_updates_task_message(self, mock_get):
        task_repo = mock.MagicMock()
        image_import = import_flow._ImportToStore(TASK_ID1, TASK_TYPE, task_repo, mock.MagicMock(), 'http://url', 'store1', False, True)
        task = mock.MagicMock()
        task.status = 'processing'
        mock_get.return_value = task
        action = mock.MagicMock()
        image_import._status_callback(action, 128, 256 * units.Mi)
        mock_get.assert_called_once_with(task_repo, TASK_ID1)
        task_repo.save.assert_called_once_with(task)
        self.assertEqual(_('Copied %i MiB' % 256), task.message)

    @mock.patch('glance.common.scripts.utils.get_task')
    def test_status_aborts_missing_task(self, mock_get):
        task_repo = mock.MagicMock()
        image_import = import_flow._ImportToStore(TASK_ID1, TASK_TYPE, task_repo, mock.MagicMock(), 'http://url', 'store1', False, True)
        mock_get.return_value = None
        action = mock.MagicMock()
        self.assertRaises(exception.TaskNotFound, image_import._status_callback, action, 128, 256 * units.Mi)
        mock_get.assert_called_once_with(task_repo, TASK_ID1)
        task_repo.save.assert_not_called()

    @mock.patch('glance.common.scripts.utils.get_task')
    def test_status_aborts_invalid_task_state(self, mock_get):
        task_repo = mock.MagicMock()
        image_import = import_flow._ImportToStore(TASK_ID1, TASK_TYPE, task_repo, mock.MagicMock(), 'http://url', 'store1', False, True)
        task = mock.MagicMock()
        task.status = 'failed'
        mock_get.return_value = task
        action = mock.MagicMock()
        self.assertRaises(exception.TaskAbortedError, image_import._status_callback, action, 128, 256 * units.Mi)
        mock_get.assert_called_once_with(task_repo, TASK_ID1)
        task_repo.save.assert_not_called()