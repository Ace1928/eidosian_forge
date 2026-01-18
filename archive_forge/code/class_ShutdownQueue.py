import os
import math
import functools
import logging
import socket
import threading
import random
import string
import concurrent.futures
from botocore.compat import six
from botocore.vendored.requests.packages.urllib3.exceptions import \
from botocore.exceptions import IncompleteReadError
import s3transfer.compat
from s3transfer.exceptions import RetriesExceededError, S3UploadFailedError
class ShutdownQueue(queue.Queue):
    """A queue implementation that can be shutdown.

    Shutting down a queue means that this class adds a
    trigger_shutdown method that will trigger all subsequent
    calls to put() to fail with a ``QueueShutdownError``.

    It purposefully deviates from queue.Queue, and is *not* meant
    to be a drop in replacement for ``queue.Queue``.

    """

    def _init(self, maxsize):
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        return queue.Queue._init(self, maxsize)

    def trigger_shutdown(self):
        with self._shutdown_lock:
            self._shutdown = True
            logger.debug('The IO queue is now shutdown.')

    def put(self, item):
        with self._shutdown_lock:
            if self._shutdown:
                raise QueueShutdownError('Cannot put item to queue when queue has been shutdown.')
        return queue.Queue.put(self, item)