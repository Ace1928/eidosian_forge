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
class ConcurrentTransferer(object):

    def __init__(self, part_size=DEFAULT_PART_SIZE, num_threads=10):
        self._part_size = part_size
        self._num_threads = num_threads
        self._threads = []

    def _calculate_required_part_size(self, total_size):
        min_part_size_required = minimum_part_size(total_size)
        if self._part_size >= min_part_size_required:
            part_size = self._part_size
        else:
            part_size = min_part_size_required
            log.debug('The part size specified (%s) is smaller than the minimum required part size.  Using a part size of: %s', self._part_size, part_size)
        total_parts = int(math.ceil(total_size / float(part_size)))
        return (total_parts, part_size)

    def _shutdown_threads(self):
        log.debug('Shutting down threads.')
        for thread in self._threads:
            thread.should_continue = False
        for thread in self._threads:
            thread.join()
        log.debug('Threads have exited.')

    def _add_work_items_to_queue(self, total_parts, worker_queue, part_size):
        log.debug('Adding work items to queue.')
        for i in range(total_parts):
            worker_queue.put((i, part_size))
        for i in range(self._num_threads):
            worker_queue.put(_END_SENTINEL)