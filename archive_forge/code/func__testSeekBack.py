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
def _testSeekBack(self, initial_position, seek_back_amount):
    """Tests reading then seeking backwards.

    This function simulates an upload that is resumed after a connection break.
    It reads one transfer buffer at a time until it reaches initial_position,
    then seeks backwards (as if the server did not receive some of the bytes)
    and reads to the end of the file, ensuring the hash matches the original
    file upon completion.

    Args:
      initial_position: Initial number of bytes to read before seek.
      seek_back_amount: Number of bytes to seek backward.

    Raises:
      AssertionError on wrong amount of data remaining or hash mismatch.
    """
    tmp_file = self._GetTestFile()
    tmp_file_len = os.path.getsize(tmp_file)
    self.assertGreaterEqual(initial_position, seek_back_amount, 'seek_back_amount must be less than initial position %s (but was actually: %s)' % (initial_position, seek_back_amount))
    self.assertLess(initial_position, tmp_file_len, 'initial_position must be less than test file size %s (but was actually: %s)' % (tmp_file_len, initial_position))
    digesters = {'md5': GetMd5()}
    with open(tmp_file, 'rb') as stream:
        wrapper = HashingFileUploadWrapper(stream, digesters, {'md5': GetMd5}, self._dummy_url, self.logger)
        position = 0
        while position < initial_position - TRANSFER_BUFFER_SIZE:
            data = wrapper.read(TRANSFER_BUFFER_SIZE)
            position += len(data)
        wrapper.read(initial_position - position)
        wrapper.seek(initial_position - seek_back_amount)
        self.assertEqual(wrapper.tell(), initial_position - seek_back_amount)
        data = wrapper.read()
        self.assertEqual(len(data), tmp_file_len - (initial_position - seek_back_amount))
    with open(tmp_file, 'rb') as stream:
        actual = CalculateMd5FromContents(stream)
    self.assertEqual(actual, digesters['md5'].hexdigest())