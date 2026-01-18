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
def _test_cinder_get(self, is_multi_store=False):
    expected_size = 5 * units.Ki
    expected_file_contents = b'*' * expected_size
    volume_file = io.BytesIO(expected_file_contents)
    fake_client = mock.MagicMock(auth_token=None, management_url=None)
    fake_volume_uuid = str(uuid.uuid4())
    fake_volume = mock.MagicMock(id=fake_volume_uuid, metadata={'image_size': expected_size}, status='available')
    fake_volume.manager.get.return_value = fake_volume
    fake_volumes = mock.MagicMock(get=lambda id: fake_volume)

    @contextlib.contextmanager
    def fake_open(client, volume, mode):
        self.assertEqual('rb', mode)
        yield volume_file
    with mock.patch.object(cinder.Store, 'get_cinderclient') as mock_cc, mock.patch.object(self.store, '_open_cinder_volume', side_effect=fake_open):
        mock_cc.return_value = mock.MagicMock(client=fake_client, volumes=fake_volumes)
        loc = self._get_uri_loc(fake_volume_uuid, is_multi_store=is_multi_store)
        image_file, image_size = self.store.get(loc, context=self.context)
        expected_num_chunks = 2
        data = b''
        num_chunks = 0
        for chunk in image_file:
            num_chunks += 1
            data += chunk
        self.assertEqual(expected_num_chunks, num_chunks)
        self.assertEqual(expected_file_contents, data)