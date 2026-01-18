import io
from unittest import mock
from oslo_config import cfg
from oslo_utils import units
import glance_store as store
from glance_store._drivers import rbd as rbd_store
from glance_store import exceptions
from glance_store import location as g_location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
def _fake_delete_image(target_pool, image_name, snapshot_name=None):
    self.assertEqual(self.location.pool, target_pool)
    self.assertEqual(self.location.image, image_name)
    self.assertEqual(self.location.snapshot, snapshot_name)
    self.called_commands_actual.append('delete')