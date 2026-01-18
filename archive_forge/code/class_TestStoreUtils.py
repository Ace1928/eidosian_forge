import io
import tempfile
from unittest import mock
import glance_store as store
from glance_store._drivers import cinder
from oslo_config import cfg
from oslo_log import log as logging
import webob
from glance.common import exception
from glance.common import store_utils
from glance.common import utils
from glance.tests.unit import base
from glance.tests import utils as test_utils
class TestStoreUtils(test_utils.BaseTestCase):
    """Test glance.common.store_utils module"""

    def _test_update_store_in_location(self, metadata, store_id, expected, store_id_call_count=1, save_call_count=1):
        image = mock.Mock()
        image_repo = mock.Mock()
        image_repo.save = mock.Mock()
        context = mock.Mock()
        locations = [{'url': 'rbd://aaaaaaaa/images/id', 'metadata': metadata}]
        image.locations = locations
        with mock.patch.object(store_utils, '_get_store_id_from_uri') as mock_get_store_id:
            mock_get_store_id.return_value = store_id
            store_utils.update_store_in_locations(context, image, image_repo)
            self.assertEqual(image.locations[0]['metadata'].get('store'), expected)
            self.assertEqual(store_id_call_count, mock_get_store_id.call_count)
            self.assertEqual(save_call_count, image_repo.save.call_count)

    def test_update_store_location_with_no_store(self):
        enabled_backends = {'rbd1': 'rbd', 'rbd2': 'rbd'}
        self.config(enabled_backends=enabled_backends)
        self._test_update_store_in_location({}, 'rbd1', 'rbd1')

    def test_update_store_location_with_different_store(self):
        enabled_backends = {'ceph1': 'rbd', 'ceph2': 'rbd'}
        self.config(enabled_backends=enabled_backends)
        self._test_update_store_in_location({'store': 'rbd2'}, 'ceph1', 'ceph1')

    def test_update_store_location_with_same_store(self):
        enabled_backends = {'rbd1': 'rbd', 'rbd2': 'rbd'}
        self.config(enabled_backends=enabled_backends)
        self._test_update_store_in_location({'store': 'rbd1'}, 'rbd1', 'rbd1', store_id_call_count=0, save_call_count=0)

    def test_update_store_location_with_store_none(self):
        enabled_backends = {'rbd1': 'rbd', 'rbd2': 'rbd'}
        self.config(enabled_backends=enabled_backends)
        self._test_update_store_in_location({}, None, None, save_call_count=0)