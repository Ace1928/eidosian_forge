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
def _set_done(self):
    self._done = True
    for done_callback in self._done_callbacks:
        self._invoke_done_callback(done_callback)
    self._done_callbacks = []