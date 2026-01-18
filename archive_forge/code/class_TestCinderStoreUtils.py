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
class TestCinderStoreUtils(base.MultiStoreClearingUnitTest):
    """Test glance.common.store_utils module for cinder multistore"""

    @mock.patch.object(cinder.Store, 'is_image_associated_with_store')
    @mock.patch.object(cinder.Store, 'url_prefix', new_callable=mock.PropertyMock)
    def _test_update_cinder_store_in_location(self, mock_url_prefix, mock_associate_store, is_valid=True):
        volume_id = 'db457a25-8f16-4b2c-a644-eae8d17fe224'
        store_id = 'fast-cinder'
        expected = 'fast-cinder'
        image = mock.Mock()
        image_repo = mock.Mock()
        image_repo.save = mock.Mock()
        context = mock.Mock()
        mock_associate_store.return_value = is_valid
        locations = [{'url': 'cinder://%s' % volume_id, 'metadata': {}}]
        mock_url_prefix.return_value = 'cinder://%s' % store_id
        image.locations = locations
        store_utils.update_store_in_locations(context, image, image_repo)
        if is_valid:
            expected_url = mock_url_prefix.return_value + '/' + volume_id
            self.assertEqual(expected_url, image.locations[0].get('url'))
            self.assertEqual(expected, image.locations[0]['metadata'].get('store'))
            self.assertEqual(1, image_repo.save.call_count)
        else:
            self.assertEqual(locations[0]['url'], image.locations[0].get('url'))
            self.assertEqual({}, image.locations[0]['metadata'])
            self.assertEqual(0, image_repo.save.call_count)

    def test_update_cinder_store_location_valid_type(self):
        self._test_update_cinder_store_in_location()

    def test_update_cinder_store_location_invalid_type(self):
        self._test_update_cinder_store_in_location(is_valid=False)