import collections
import contextlib
import logging
import multiprocessing
import threading
import signal
from copy import deepcopy
import botocore.session
from botocore.config import Config
from s3transfer.constants import MB
from s3transfer.constants import ALLOWED_DOWNLOAD_ARGS
from s3transfer.constants import PROCESS_USER_AGENT
from s3transfer.compat import MAXINT
from s3transfer.compat import BaseManager
from s3transfer.exceptions import CancelledError
from s3transfer.exceptions import RetriesExceededError
from s3transfer.futures import BaseTransferFuture
from s3transfer.futures import BaseTransferMeta
from s3transfer.utils import S3_RETRYABLE_DOWNLOAD_ERRORS
from s3transfer.utils import calculate_num_parts
from s3transfer.utils import calculate_range_parameter
from s3transfer.utils import OSUtils
from s3transfer.utils import CallArgs
class TransferState(object):
    """Represents the current state of an individual transfer"""

    def __init__(self):
        self._exception = None
        self._done_event = threading.Event()
        self._job_lock = threading.Lock()
        self._jobs_to_complete = 0

    @property
    def done(self):
        return self._done_event.is_set()

    def set_done(self):
        self._done_event.set()

    def wait_till_done(self):
        self._done_event.wait(MAXINT)

    @property
    def exception(self):
        return self._exception

    @exception.setter
    def exception(self, val):
        self._exception = val

    @property
    def jobs_to_complete(self):
        return self._jobs_to_complete

    @jobs_to_complete.setter
    def jobs_to_complete(self, val):
        self._jobs_to_complete = val

    def decrement_jobs_to_complete(self):
        with self._job_lock:
            self._jobs_to_complete -= 1
            return self._jobs_to_complete