import datetime
import errno
import io
import os
import time
from unittest import mock
from oslo_utils import fileutils
from glance.image_cache.drivers import centralized_db
from glance.tests import functional
def _test_clean_stall_time(self, stall_time=None, days=2, stall_failed=False):
    """
        Test the clean method removes the stalled images as expected
        """
    self.start_server(enable_cache=True)
    self.driver = centralized_db.Driver()
    self.driver.configure()
    cache_dir = os.path.join(self.test_dir, 'cache')
    incomplete_file_path_1 = os.path.join(cache_dir, 'incomplete', '1')
    incomplete_file_path_2 = os.path.join(cache_dir, 'incomplete', '2')
    for f in (incomplete_file_path_1, incomplete_file_path_2):
        incomplete_file = open(f, 'wb')
        incomplete_file.write(DATA)
        incomplete_file.close()
    mtime = os.path.getmtime(incomplete_file_path_1)
    pastday = datetime.datetime.fromtimestamp(mtime) - datetime.timedelta(days=days)
    atime = int(time.mktime(pastday.timetuple()))
    mtime = atime
    os.utime(incomplete_file_path_1, (atime, mtime))
    self.assertTrue(os.path.exists(incomplete_file_path_1))
    self.assertTrue(os.path.exists(incomplete_file_path_2))
    if stall_failed:
        with mock.patch.object(fileutils, 'delete_if_exists') as mock_delete:
            mock_delete.side_effect = OSError(errno.ENOENT, '')
            self.driver.clean(stall_time=stall_time)
            self.assertTrue(os.path.exists(incomplete_file_path_1))
    else:
        self.driver.clean(stall_time=stall_time)
        self.assertFalse(os.path.exists(incomplete_file_path_1))
    self.assertTrue(os.path.exists(incomplete_file_path_2))