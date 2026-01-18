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
class TestImportActionWrapper(test_utils.BaseTestCase):

    def test_wrapper_success(self):
        mock_repo = mock.MagicMock()
        mock_repo.get.return_value.extra_properties = {'os_glance_import_task': TASK_ID1}
        wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, TASK_ID1)
        with wrapper as action:
            self.assertIsInstance(action, import_flow._ImportActions)
        mock_repo.get.assert_has_calls([mock.call(IMAGE_ID1), mock.call(IMAGE_ID1)])
        mock_repo.save.assert_called_once_with(mock_repo.get.return_value, mock_repo.get.return_value.status)

    def test_wrapper_failure(self):
        mock_repo = mock.MagicMock()
        mock_repo.get.return_value.extra_properties = {'os_glance_import_task': TASK_ID1}
        wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, TASK_ID1)

        class SpecificError(Exception):
            pass
        try:
            with wrapper:
                raise SpecificError('some failure')
        except SpecificError:
            pass
        mock_repo.get.assert_called_once_with(IMAGE_ID1)
        mock_repo.save.assert_not_called()

    @mock.patch.object(import_flow, 'LOG')
    def test_wrapper_logs_status(self, mock_log):
        mock_repo = mock.MagicMock()
        mock_image = mock_repo.get.return_value
        mock_image.extra_properties = {'os_glance_import_task': TASK_ID1}
        wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, TASK_ID1)
        mock_image.status = 'foo'
        with wrapper as action:
            action.set_image_attribute(status='bar')
        mock_log.debug.assert_called_once_with('Image %(image_id)s status changing from %(old_status)s to %(new_status)s', {'image_id': IMAGE_ID1, 'old_status': 'foo', 'new_status': 'bar'})
        self.assertEqual('bar', mock_image.status)

    def test_image_id_property(self):
        mock_repo = mock.MagicMock()
        wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, TASK_ID1)
        self.assertEqual(IMAGE_ID1, wrapper.image_id)

    def test_set_image_attribute(self):
        mock_repo = mock.MagicMock()
        mock_image = mock_repo.get.return_value
        mock_image.extra_properties = {'os_glance_import_task': TASK_ID1}
        mock_image.status = 'bar'
        wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, TASK_ID1)
        with wrapper as action:
            action.set_image_attribute(status='foo', virtual_size=123, size=64)
        mock_repo.save.assert_called_once_with(mock_image, 'bar')
        self.assertEqual('foo', mock_image.status)
        self.assertEqual(123, mock_image.virtual_size)
        self.assertEqual(64, mock_image.size)

    def test_set_image_attribute_disallowed(self):
        mock_repo = mock.MagicMock()
        mock_image = mock_repo.get.return_value
        mock_image.extra_properties = {'os_glance_import_task': TASK_ID1}
        mock_image.status = 'bar'
        wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, TASK_ID1)
        with wrapper as action:
            self.assertRaises(AttributeError, action.set_image_attribute, id='foo')

    @mock.patch.object(import_flow, 'LOG')
    def test_set_image_extra_properties(self, mock_log):
        mock_repo = mock.MagicMock()
        mock_image = mock_repo.get.return_value
        mock_image.image_id = IMAGE_ID1
        mock_image.extra_properties = {'os_glance_import_task': TASK_ID1}
        mock_image.status = 'bar'
        wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, TASK_ID1)
        with wrapper as action:
            action.set_image_extra_properties({'os_glance_foo': 'bar'})
        self.assertEqual({'os_glance_import_task': TASK_ID1}, mock_image.extra_properties)
        mock_log.warning.assert_called()
        mock_log.warning.reset_mock()
        with wrapper as action:
            action.set_image_extra_properties({'os_glance_foo': 'bar', 'os_glance_baz': 'bat'})
        self.assertEqual({'os_glance_import_task': TASK_ID1}, mock_image.extra_properties)
        mock_log.warning.assert_called()
        mock_log.warning.reset_mock()
        with wrapper as action:
            action.set_image_extra_properties({'foo': 'bar', 'os_glance_foo': 'baz'})
        self.assertEqual({'foo': 'bar', 'os_glance_import_task': TASK_ID1}, mock_image.extra_properties)
        mock_log.warning.assert_called_once_with('Dropping %(key)s=%(val)s during metadata injection for %(image)s', {'key': 'os_glance_foo', 'val': 'baz', 'image': IMAGE_ID1})

    def test_image_size(self):
        mock_repo = mock.MagicMock()
        mock_image = mock_repo.get.return_value
        mock_image.image_id = IMAGE_ID1
        mock_image.extra_properties = {'os_glance_import_task': TASK_ID1}
        mock_image.size = 123
        wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, TASK_ID1)
        with wrapper as action:
            self.assertEqual(123, action.image_size)

    def test_image_locations(self):
        mock_repo = mock.MagicMock()
        mock_image = mock_repo.get.return_value
        mock_image.image_id = IMAGE_ID1
        mock_image.extra_properties = {'os_glance_import_task': TASK_ID1}
        mock_image.locations = {'some': {'complex': ['structure']}}
        wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, TASK_ID1)
        with wrapper as action:
            self.assertEqual({'some': {'complex': ['structure']}}, action.image_locations)
            action.image_locations['foo'] = 'bar'
        self.assertEqual({'some': {'complex': ['structure']}}, mock_image.locations)

    def test_drop_lock_for_task(self):
        mock_repo = mock.MagicMock()
        mock_repo.get.return_value.extra_properties = {'os_glance_import_task': TASK_ID1}
        wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, TASK_ID1)
        wrapper.drop_lock_for_task()
        mock_repo.delete_property_atomic.assert_called_once_with(mock_repo.get.return_value, 'os_glance_import_task', TASK_ID1)

    def test_assert_task_lock(self):
        mock_repo = mock.MagicMock()
        mock_repo.get.return_value.extra_properties = {'os_glance_import_task': TASK_ID1}
        wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, TASK_ID1)
        wrapper.assert_task_lock()
        wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, 'foo')
        self.assertRaises(exception.TaskAbortedError, wrapper.assert_task_lock)

    def _grab_image(self, wrapper):
        with wrapper:
            pass

    @mock.patch.object(import_flow, 'LOG')
    def test_check_task_lock(self, mock_log):
        mock_repo = mock.MagicMock()
        wrapper = import_flow.ImportActionWrapper(mock_repo, IMAGE_ID1, TASK_ID1)
        image = mock.MagicMock(image_id=IMAGE_ID1)
        image.extra_properties = {'os_glance_import_task': TASK_ID1}
        mock_repo.get.return_value = image
        self._grab_image(wrapper)
        mock_log.error.assert_not_called()
        image.extra_properties['os_glance_import_task'] = 'somethingelse'
        self.assertRaises(exception.TaskAbortedError, self._grab_image, wrapper)
        mock_log.error.assert_called_once_with('Image %(image)s import task %(task)s attempted to take action on image, but other task %(other)s holds the lock; Aborting.', {'image': image.image_id, 'task': TASK_ID1, 'other': 'somethingelse'})