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
def _calculate_required_part_size(self, total_size):
    min_part_size_required = minimum_part_size(total_size)
    if self._part_size >= min_part_size_required:
        part_size = self._part_size
    else:
        part_size = min_part_size_required
        log.debug('The part size specified (%s) is smaller than the minimum required part size.  Using a part size of: %s', self._part_size, part_size)
    total_parts = int(math.ceil(total_size / float(part_size)))
    return (total_parts, part_size)