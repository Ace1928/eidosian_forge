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
def _adjust_for_max_parts(self, current_chunksize, file_size):
    chunksize = current_chunksize
    num_parts = int(math.ceil(file_size / float(chunksize)))
    while num_parts > self.max_parts:
        chunksize *= 2
        num_parts = int(math.ceil(file_size / float(chunksize)))
    if chunksize != current_chunksize:
        logger.debug('Chunksize would result in the number of parts exceeding the maximum. Setting to %s from %s.' % (chunksize, current_chunksize))
    return chunksize