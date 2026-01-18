import contextlib
import hashlib
import io
import math
import os
from unittest import mock
import socket
import sys
import tempfile
import time
import uuid
from keystoneauth1 import exceptions as keystone_exc
from os_brick.initiator import connector
from oslo_concurrency import processutils
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers.cinder import scaleio
from glance_store.common import attachment_state_manager
from glance_store.common import cinder_utils
from glance_store import exceptions
from glance_store import location
from glance_store._drivers.cinder import store as cinder # noqa
from glance_store._drivers.cinder import nfs # noqa
def _test_cinder_volume_not_found(self, method_call, mock_method):
    fake_volume_uuid = str(uuid.uuid4())
    loc = mock.MagicMock(volume_id=fake_volume_uuid)
    mock_not_found = {mock_method: mock.MagicMock(side_effect=cinder.cinder_exception.NotFound(code=404))}
    fake_volumes = mock.MagicMock(**mock_not_found)
    with mock.patch.object(cinder.Store, 'get_cinderclient') as mocked_cc:
        mocked_cc.return_value = mock.MagicMock(volumes=fake_volumes)
        self.assertRaises(exceptions.NotFound, method_call, loc, context=self.context)