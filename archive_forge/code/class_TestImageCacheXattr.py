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
class TestImageCacheXattr(test_utils.BaseTestCase, ImageCacheTestCase):
    """Tests image caching when xattr is used in cache"""

    def setUp(self):
        """
        Test to see if the pre-requisites for the image cache
        are working (python-xattr installed and xattr support on the
        filesystem)
        """
        super(TestImageCacheXattr, self).setUp()
        if getattr(self, 'disable', False):
            return
        self.cache_dir = self.useFixture(fixtures.TempDir()).path
        if not getattr(self, 'inited', False):
            try:
                import xattr
            except ImportError:
                self.inited = True
                self.disabled = True
                self.disabled_message = 'python-xattr not installed.'
                return
        self.inited = True
        self.disabled = False
        self.config(image_cache_dir=self.cache_dir, image_cache_driver='xattr', image_cache_max_size=5 * units.Ki)
        self.cache = image_cache.ImageCache()
        if not xattr_writes_supported(self.cache_dir):
            self.inited = True
            self.disabled = True
            self.disabled_message = 'filesystem does not support xattr'
            return