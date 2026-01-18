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
@mock.patch('glance.async_.flows.api_image_import._VerifyStaging.__init__')
@mock.patch('taskflow.patterns.linear_flow.Flow.add')
@mock.patch('taskflow.patterns.linear_flow.__init__')
def _pass_uri(self, mock_lf_init, mock_flow_add, mock_VS_init, uri, file_uri, import_req):
    flow_kwargs = {'task_id': TASK_ID1, 'task_type': TASK_TYPE, 'task_repo': self.mock_task_repo, 'image_repo': self.mock_image_repo, 'image_id': IMAGE_ID1, 'context': mock.MagicMock(), 'import_req': import_req}
    mock_lf_init.return_value = None
    mock_VS_init.return_value = None
    self.config(node_staging_uri=uri)
    import_flow.get_flow(**flow_kwargs)
    mock_VS_init.assert_called_with(TASK_ID1, TASK_TYPE, self.mock_task_repo, file_uri)