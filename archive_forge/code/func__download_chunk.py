import os
import math
import threading
import hashlib
import time
import logging
from boto.compat import Queue
import binascii
from boto.glacier.utils import DEFAULT_PART_SIZE, minimum_part_size, \
from boto.glacier.exceptions import UploadArchiveError, \
def _download_chunk(self, work):
    """
        Downloads a chunk of archive from Glacier. Saves the data to a temp file
        Returns the part number and temp file location

        :param work:
        """
    part_number, part_size = work
    start_byte = part_number * part_size
    byte_range = (start_byte, start_byte + part_size - 1)
    log.debug('Downloading chunk %s of size %s', part_number, part_size)
    response = self._job.get_output(byte_range)
    data = response.read()
    actual_hash = bytes_to_hex(tree_hash(chunk_hashes(data)))
    if response['TreeHash'] != actual_hash:
        raise TreeHashDoesNotMatchError('Tree hash for part number %s does not match, expected: %s, got: %s' % (part_number, response['TreeHash'], actual_hash))
    return (part_number, part_size, binascii.unhexlify(actual_hash), data)