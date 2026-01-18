import os
from unittest import mock
import glance_store as store
from glance_store._drivers import cinder
from glance_store._drivers import rbd as rbd_store
from glance_store._drivers import swift
from glance_store import location
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_db import options
from oslo_serialization import jsonutils
from glance.tests import stubs
from glance.tests import utils as test_utils
def _create_multi_stores(self, passing_config=True):
    """Create known stores.

        :param passing_config: making store driver passes basic configurations.
        :returns: the number of how many store drivers been loaded.
        """
    rbd_store.rados = mock.MagicMock()
    rbd_store.rbd = mock.MagicMock()
    rbd_store.Store._set_url_prefix = mock.MagicMock()
    cinder.cinderclient = mock.MagicMock()
    cinder.Store.get_cinderclient = mock.MagicMock()
    swift.swiftclient = mock.MagicMock()
    swift.BaseStore.get_store_connection = mock.MagicMock()
    self.config(enabled_backends={'fast': 'file', 'cheap': 'file', 'readonly_store': 'http', 'fast-cinder': 'cinder', 'fast-rbd': 'rbd', 'reliable': 'swift'})
    store.register_store_opts(CONF)
    self.config(default_backend='fast', group='glance_store')
    self.config(filesystem_store_datadir=self.test_dir, filesystem_thin_provisioning=False, filesystem_store_chunk_size=65536, group='fast')
    self.config(filesystem_store_datadir=self.test_dir2, filesystem_thin_provisioning=False, filesystem_store_chunk_size=65536, group='cheap')
    self.config(rbd_store_chunk_size=8688388, rbd_store_pool='images', rbd_thin_provisioning=False, group='fast-rbd')
    self.config(cinder_volume_type='lvmdriver-1', cinder_use_multipath=False, group='fast-cinder')
    self.config(swift_store_container='glance', swift_store_large_object_size=524288000, swift_store_large_object_chunk_size=204800000, group='reliable')
    store.create_multi_stores(CONF)