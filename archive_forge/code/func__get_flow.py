from unittest import mock
import futurist
import glance_store as store
from oslo_config import cfg
from taskflow.patterns import linear_flow
import glance.async_
from glance.async_.flows import api_image_import
import glance.tests.utils as test_utils
def _get_flow(self, import_req=None):
    inputs = {'task_id': mock.sentinel.task_id, 'task_type': mock.MagicMock(), 'task_repo': mock.MagicMock(), 'image_repo': mock.MagicMock(), 'image_id': mock.MagicMock(), 'import_req': import_req or mock.MagicMock(), 'context': mock.MagicMock()}
    inputs['image_repo'].get.return_value = mock.MagicMock(extra_properties={'os_glance_import_task': mock.sentinel.task_id})
    flow = api_image_import.get_flow(**inputs)
    return flow