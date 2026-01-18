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
def _test_clean_invalid_path(self, failure=False):
    invalid_file_path = os.path.join(self.cache_dir, 'invalid', '1')
    invalid_file = open(invalid_file_path, 'wb')
    invalid_file.write(FIXTURE_DATA)
    invalid_file.close()
    self.assertTrue(os.path.exists(invalid_file_path))
    self.delay_inaccurate_clock()
    if failure:
        with mock.patch.object(fileutils, 'delete_if_exists') as mock_delete:
            mock_delete.side_effect = OSError(errno.ENOENT, '')
            try:
                self.cache.clean()
            except OSError:
                self.assertTrue(os.path.exists(invalid_file_path))
    else:
        self.cache.clean()
        self.assertFalse(os.path.exists(invalid_file_path))