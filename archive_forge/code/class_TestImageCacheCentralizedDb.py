from contextlib import contextmanager
import datetime
import errno
import io
import os
import tempfile
import time
from unittest import mock
import fixtures
import glance_store as store
from oslo_config import cfg
from oslo_utils import fileutils
from oslo_utils import secretutils
from oslo_utils import units
from glance import async_
from glance.common import exception
from glance import context
from glance import gateway as glance_gateway
from glance import image_cache
from glance.image_cache import prefetcher
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
from glance.tests.utils import skip_if_disabled
from glance.tests.utils import xattr_writes_supported
class TestImageCacheCentralizedDb(test_utils.BaseTestCase, ImageCacheTestCase):
    """Tests image caching when Centralized DB is used in cache"""

    def setUp(self):
        super(TestImageCacheCentralizedDb, self).setUp()
        self.inited = True
        self.disabled = False
        self.cache_dir = self.useFixture(fixtures.TempDir()).path
        self.config(image_cache_dir=self.cache_dir, image_cache_driver='centralized_db', image_cache_max_size=5 * units.Ki, worker_self_reference_url='http://workerx')
        with mock.patch('glance.db.get_api') as mock_get_db:
            self.db = unit_test_utils.FakeDB(initialize=False)
            mock_get_db.return_value = self.db
            self.cache = image_cache.ImageCache()

    def test_node_reference_create_duplicate(self):
        with mock.patch('glance.db.get_api') as mock_get_db:
            self.db = unit_test_utils.FakeDB(initialize=False)
            mock_get_db.return_value = self.db
            with mock.patch.object(self.db, 'node_reference_create') as mock_node_create:
                mock_node_create.side_effect = exception.Duplicate
                with mock.patch.object(image_cache.drivers.centralized_db, 'LOG') as mock_log:
                    image_cache.ImageCache()
                    expected_calls = [mock.call('Node reference is already recorded, ignoring it')]
                    mock_log.debug.assert_has_calls(expected_calls)

    def test_get_least_recently_accessed_os_error(self):
        self.assertEqual(0, self.cache.get_cache_size())
        for x in range(10):
            FIXTURE_FILE = io.BytesIO(FIXTURE_DATA)
            self.assertTrue(self.cache.cache_image_file(x, FIXTURE_FILE))
        self.assertEqual(10 * units.Ki, self.cache.get_cache_size())
        with mock.patch.object(os, 'stat') as mock_stat:
            mock_stat.side_effect = OSError
            image_id, size = self.cache.driver.get_least_recently_accessed()
            self.assertEqual(0, size)

    @skip_if_disabled
    def test_clean_stalled_fails(self):
        """Test the clean method fails to delete file, ignores the failure"""
        self._test_clean_stall_time(stall_time=3600, days=1, stall_failed=True)

    @skip_if_disabled
    def test_clean_invalid_path_fails(self):
        """Test the clean method fails to remove image from invalid path."""
        self._test_clean_invalid_path(failure=True)