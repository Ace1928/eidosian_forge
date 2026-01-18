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
def _test_cinder_add(self, fake_volume, volume_file, size_kb=5, verifier=None, backend='glance_store', is_multi_store=False):
    expected_image_id = str(uuid.uuid4())
    expected_size = size_kb * units.Ki
    expected_file_contents = b'*' * expected_size
    image_file = io.BytesIO(expected_file_contents)
    expected_checksum = md5(expected_file_contents, usedforsecurity=False).hexdigest()
    expected_multihash = hashlib.sha256(expected_file_contents).hexdigest()
    expected_location = 'cinder://%s' % fake_volume.id
    if is_multi_store:
        if backend == 'glance_store':
            backend = 'cinder1'
        expected_location = 'cinder://%s/%s' % (backend, fake_volume.id)
    self.config(cinder_volume_type='some_type', group=backend)
    fake_client = mock.MagicMock(auth_token=None, management_url=None)
    fake_volume.manager.get.return_value = fake_volume
    fake_volumes = mock.MagicMock(create=mock.Mock(return_value=fake_volume))

    @contextlib.contextmanager
    def fake_open(client, volume, mode):
        self.assertEqual('wb', mode)
        yield volume_file
    with mock.patch.object(cinder.Store, 'get_cinderclient') as mock_cc, mock.patch.object(self.store, '_open_cinder_volume', side_effect=fake_open):
        mock_cc.return_value = mock.MagicMock(client=fake_client, volumes=fake_volumes)
        loc, size, checksum, multihash, metadata = self.store.add(expected_image_id, image_file, expected_size, self.hash_algo, self.context, verifier)
        self.assertEqual(expected_location, loc)
        self.assertEqual(expected_size, size)
        self.assertEqual(expected_checksum, checksum)
        self.assertEqual(expected_multihash, multihash)
        fake_volumes.create.assert_called_once_with(1, name='image-%s' % expected_image_id, metadata={'image_owner': self.context.project_id, 'glance_image_id': expected_image_id, 'image_size': str(expected_size)}, volume_type='some_type')
        if is_multi_store:
            self.assertEqual(backend, metadata['store'])