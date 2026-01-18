import os
import tempfile
from os_win import constants
from os_win.tests.functional import test_base
from os_win import utilsfactory
def _check_attached(expect_attached):
    paths = [vhd_path, vhd_link, vhd_link2]
    for path in paths:
        self.assertEqual(expect_attached, self._vhdutils.is_virtual_disk_file_attached(path))
        self.assertEqual(expect_attached, self._diskutils.is_virtual_disk_file_attached(path))