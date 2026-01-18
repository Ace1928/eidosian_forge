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
def _test_cinder_delete(self, is_multi_store=False):
    fake_client = mock.MagicMock(auth_token=None, management_url=None)
    fake_volume_uuid = str(uuid.uuid4())
    fake_volumes = mock.MagicMock(delete=mock.Mock())
    with mock.patch.object(cinder.Store, 'get_cinderclient') as mocked_cc:
        mocked_cc.return_value = mock.MagicMock(client=fake_client, volumes=fake_volumes)
        loc = self._get_uri_loc(fake_volume_uuid, is_multi_store=is_multi_store)
        self.store.delete(loc, context=self.context)
        fake_volumes.delete.assert_called_once_with(fake_volume_uuid)