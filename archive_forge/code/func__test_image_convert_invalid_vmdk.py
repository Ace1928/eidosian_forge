import json
import os
from unittest import mock
import glance_store
from oslo_concurrency import processutils
from oslo_config import cfg
import glance.async_.flows.api_image_import as import_flow
import glance.async_.flows.plugins.image_conversion as image_conversion
from glance.async_ import utils as async_utils
from glance.common import utils
from glance import domain
from glance import gateway
import glance.tests.utils as test_utils
def _test_image_convert_invalid_vmdk(self):
    data = {'format': 'vmdk', 'format-specific': {'data': {'create-type': 'monolithicFlat'}}}
    convert = self._setup_image_convert_info_fail()
    with mock.patch.object(processutils, 'execute') as exc_mock:
        exc_mock.return_value = (json.dumps(data), '')
        convert.execute('file:///test/path.vmdk')