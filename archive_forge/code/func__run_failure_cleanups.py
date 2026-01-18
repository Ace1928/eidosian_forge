from concurrent import futures
from collections import namedtuple
import copy
import logging
import sys
import threading
from s3transfer.compat import MAXINT
from s3transfer.compat import six
from s3transfer.exceptions import CancelledError, TransferNotDoneError
from s3transfer.utils import FunctionContainer
from s3transfer.utils import TaskSemaphore
def _run_failure_cleanups(self):
    with self._failure_cleanups_lock:
        self._run_callbacks(self.failure_cleanups)
        self._failure_cleanups = []