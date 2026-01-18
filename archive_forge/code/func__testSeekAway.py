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
def _testSeekAway(self, initial_read):
    """Tests reading to an initial position and then seeking to EOF and back.

    This function simulates an size check on the input file by seeking to the
    end of the file and then back to the current position. Then it reads to
    the end of the file, ensuring the hash matches the original file upon
    completion.

    Args:
      initial_read: Number of bytes to initially read.

    Raises:
      AssertionError on wrong amount of data remaining or hash mismatch.
    """
    tmp_file = self._GetTestFile()
    tmp_file_len = os.path.getsize(tmp_file)
    self.assertLess(initial_read, tmp_file_len, 'initial_read must be less than test file size %s (but was actually: %s)' % (tmp_file_len, initial_read))
    digesters = {'md5': GetMd5()}
    with open(tmp_file, 'rb') as stream:
        wrapper = HashingFileUploadWrapper(stream, digesters, {'md5': GetMd5}, self._dummy_url, self.logger)
        wrapper.read(initial_read)
        self.assertEqual(wrapper.tell(), initial_read)
        wrapper.seek(0, os.SEEK_END)
        self.assertEqual(wrapper.tell(), tmp_file_len)
        wrapper.seek(initial_read, os.SEEK_SET)
        data = wrapper.read()
        self.assertEqual(len(data), tmp_file_len - initial_read)
    with open(tmp_file, 'rb') as stream:
        actual = CalculateMd5FromContents(stream)
    self.assertEqual(actual, digesters['md5'].hexdigest())