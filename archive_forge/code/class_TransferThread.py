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
class TransferThread(threading.Thread):

    def __init__(self, worker_queue, result_queue):
        super(TransferThread, self).__init__()
        self._worker_queue = worker_queue
        self._result_queue = result_queue
        self.should_continue = True

    def run(self):
        while self.should_continue:
            try:
                work = self._worker_queue.get(timeout=1)
            except Empty:
                continue
            if work is _END_SENTINEL:
                self._cleanup()
                return
            result = self._process_chunk(work)
            self._result_queue.put(result)
        self._cleanup()

    def _process_chunk(self, work):
        pass

    def _cleanup(self):
        pass