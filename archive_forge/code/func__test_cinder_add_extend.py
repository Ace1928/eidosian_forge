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
def _test_cinder_add_extend(self, is_multi_store=False, online=False):
    expected_volume_size = 2 * units.Gi
    expected_multihash = 'fake_hash'
    fakebuffer = mock.MagicMock()
    fakebuffer.__len__.return_value = int(expected_volume_size / 2)

    def get_fake_hash(type, secure=False):
        if type == 'md5':
            return mock.MagicMock(hexdigest=lambda: expected_checksum)
        else:
            return mock.MagicMock(hexdigest=lambda: expected_multihash)
    expected_image_id = str(uuid.uuid4())
    expected_volume_id = str(uuid.uuid4())
    expected_size = 0
    image_file = mock.MagicMock(read=mock.MagicMock(side_effect=[fakebuffer, fakebuffer, None]))
    fake_volume = mock.MagicMock(id=expected_volume_id, status='available', size=1)
    expected_checksum = 'fake_checksum'
    verifier = None
    backend = 'glance_store'
    expected_location = 'cinder://%s' % fake_volume.id
    if is_multi_store:
        backend = 'cinder1'
        expected_location = 'cinder://%s/%s' % (backend, fake_volume.id)
    self.config(cinder_volume_type='some_type', group=backend)
    if online:
        self.config(cinder_do_extend_attached=True, group=backend)
        fake_connector = mock.MagicMock()
        fake_vol_connector_map = {expected_volume_id: fake_connector}
        self.store.volume_connector_map = fake_vol_connector_map
    fake_client = mock.MagicMock(auth_token=None, management_url=None)
    fake_volume.manager.get.return_value = fake_volume
    fake_volumes = mock.MagicMock(create=mock.Mock(return_value=fake_volume))

    @contextlib.contextmanager
    def fake_open(client, volume, mode):
        self.assertEqual('wb', mode)
        yield mock.MagicMock()
    with mock.patch.object(cinder.Store, 'get_cinderclient') as mock_cc, mock.patch.object(self.store, '_open_cinder_volume', side_effect=fake_open), mock.patch.object(cinder.utils, 'get_hasher') as fake_hasher, mock.patch.object(cinder.Store, '_wait_volume_status', return_value=fake_volume) as mock_wait, mock.patch.object(cinder_utils.API, 'extend_volume') as extend_vol:
        mock_cc_return_val = mock.MagicMock(client=fake_client, volumes=fake_volumes)
        mock_cc.return_value = mock_cc_return_val
        fake_hasher.side_effect = get_fake_hash
        loc, size, checksum, multihash, metadata = self.store.add(expected_image_id, image_file, expected_size, self.hash_algo, self.context, verifier)
        self.assertEqual(expected_location, loc)
        self.assertEqual(expected_volume_size, size)
        self.assertEqual(expected_checksum, checksum)
        self.assertEqual(expected_multihash, multihash)
        fake_volumes.create.assert_called_once_with(1, name='image-%s' % expected_image_id, metadata={'image_owner': self.context.project_id, 'glance_image_id': expected_image_id, 'image_size': str(expected_volume_size)}, volume_type='some_type')
        if is_multi_store:
            self.assertEqual(backend, metadata['store'])
        if online:
            extend_vol.assert_called_once_with(mock_cc_return_val, fake_volume, expected_volume_size // units.Gi)
            mock_wait.assert_has_calls([mock.call(fake_volume, 'creating', 'available'), mock.call(fake_volume, 'extending', 'in-use')])
        else:
            fake_volume.extend.assert_called_once_with(fake_volume, expected_volume_size // units.Gi)
            mock_wait.assert_has_calls([mock.call(fake_volume, 'creating', 'available'), mock.call(fake_volume, 'extending', 'available')])