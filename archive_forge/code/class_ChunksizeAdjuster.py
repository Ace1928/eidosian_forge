import random
import time
import functools
import math
import os
import socket
import stat
import string
import logging
import threading
import io
from collections import defaultdict
from botocore.exceptions import IncompleteReadError
from botocore.exceptions import ReadTimeoutError
from s3transfer.compat import SOCKET_ERROR
from s3transfer.compat import rename_file
from s3transfer.compat import seekable
from s3transfer.compat import fallocate
class ChunksizeAdjuster(object):

    def __init__(self, max_size=MAX_SINGLE_UPLOAD_SIZE, min_size=MIN_UPLOAD_CHUNKSIZE, max_parts=MAX_PARTS):
        self.max_size = max_size
        self.min_size = min_size
        self.max_parts = max_parts

    def adjust_chunksize(self, current_chunksize, file_size=None):
        """Get a chunksize close to current that fits within all S3 limits.

        :type current_chunksize: int
        :param current_chunksize: The currently configured chunksize.

        :type file_size: int or None
        :param file_size: The size of the file to upload. This might be None
            if the object being transferred has an unknown size.

        :returns: A valid chunksize that fits within configured limits.
        """
        chunksize = current_chunksize
        if file_size is not None:
            chunksize = self._adjust_for_max_parts(chunksize, file_size)
        return self._adjust_for_chunksize_limits(chunksize)

    def _adjust_for_chunksize_limits(self, current_chunksize):
        if current_chunksize > self.max_size:
            logger.debug('Chunksize greater than maximum chunksize. Setting to %s from %s.' % (self.max_size, current_chunksize))
            return self.max_size
        elif current_chunksize < self.min_size:
            logger.debug('Chunksize less than minimum chunksize. Setting to %s from %s.' % (self.min_size, current_chunksize))
            return self.min_size
        else:
            return current_chunksize

    def _adjust_for_max_parts(self, current_chunksize, file_size):
        chunksize = current_chunksize
        num_parts = int(math.ceil(file_size / float(chunksize)))
        while num_parts > self.max_parts:
            chunksize *= 2
            num_parts = int(math.ceil(file_size / float(chunksize)))
        if chunksize != current_chunksize:
            logger.debug('Chunksize would result in the number of parts exceeding the maximum. Setting to %s from %s.' % (chunksize, current_chunksize))
        return chunksize