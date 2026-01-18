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
def _test_cinder_get_size_with_metadata(self, is_multi_store=False):
    fake_client = mock.MagicMock(auth_token=None, management_url=None)
    fake_volume_uuid = str(uuid.uuid4())
    expected_image_size = 4500 * units.Mi
    fake_volume = mock.MagicMock(size=5, metadata={'image_size': expected_image_size})
    fake_volumes = {fake_volume_uuid: fake_volume}
    with mock.patch.object(cinder.Store, 'get_cinderclient') as mocked_cc:
        mocked_cc.return_value = mock.MagicMock(client=fake_client, volumes=fake_volumes)
        loc = self._get_uri_loc(fake_volume_uuid, is_multi_store=is_multi_store)
        image_size = self.store.get_size(loc, context=self.context)
        self.assertEqual(expected_image_size, image_size)