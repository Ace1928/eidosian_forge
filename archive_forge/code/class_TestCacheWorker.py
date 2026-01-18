from unittest import mock
from glance.api.v2 import cached_images
from glance import notifier
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
class TestCacheWorker(test_utils.BaseTestCase):

    @mock.patch('glance.image_cache.prefetcher.Prefetcher')
    def test_worker_lifecycle(self, mock_pf):
        worker = cached_images.CacheWorker()
        self.assertFalse(worker.is_alive())
        worker.start()
        self.assertTrue(worker.is_alive())
        worker.submit('123')
        worker.submit('456')
        self.assertTrue(worker.is_alive())
        worker.terminate()
        self.assertFalse(worker.is_alive())
        mock_pf.return_value.fetch_image_into_cache.assert_has_calls([mock.call('123'), mock.call('456')])