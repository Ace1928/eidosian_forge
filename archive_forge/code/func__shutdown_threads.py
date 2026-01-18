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
def _shutdown_threads(self):
    log.debug('Shutting down threads.')
    for thread in self._threads:
        thread.should_continue = False
    for thread in self._threads:
        thread.join()
    log.debug('Threads have exited.')