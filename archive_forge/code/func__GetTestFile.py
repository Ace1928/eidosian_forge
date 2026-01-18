from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import hashlib
import os
import pkgutil
from unittest import mock
from gslib.exception import CommandException
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.utils.constants import TRANSFER_BUFFER_SIZE
from gslib.utils.hashing_helper import CalculateMd5FromContents
from gslib.utils.hashing_helper import GetMd5
from gslib.utils.hashing_helper import HashingFileUploadWrapper
def _GetTestFile(self):
    contents = pkgutil.get_data('gslib', 'tests/test_data/%s' % _TEST_FILE)
    if not self._temp_test_file:
        self._temp_test_file = self.CreateTempFile(file_name=_TEST_FILE, contents=contents)
    return self._temp_test_file