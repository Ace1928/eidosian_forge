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
def _run_callbacks(self, callbacks):
    for callback in callbacks:
        self._run_callback(callback)