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
class NonThreadedExecutorFuture(object):
    """The Future returned from NonThreadedExecutor

    Note that this future is **not** thread-safe as it is being used
    from the context of a non-threaded environment.
    """

    def __init__(self):
        self._result = None
        self._exception = None
        self._traceback = None
        self._done = False
        self._done_callbacks = []

    def set_result(self, result):
        self._result = result
        self._set_done()

    def set_exception_info(self, exception, traceback):
        self._exception = exception
        self._traceback = traceback
        self._set_done()

    def result(self, timeout=None):
        if self._exception:
            six.reraise(type(self._exception), self._exception, self._traceback)
        return self._result

    def _set_done(self):
        self._done = True
        for done_callback in self._done_callbacks:
            self._invoke_done_callback(done_callback)
        self._done_callbacks = []

    def _invoke_done_callback(self, done_callback):
        return done_callback(self)

    def done(self):
        return self._done

    def add_done_callback(self, fn):
        if self._done:
            self._invoke_done_callback(fn)
        else:
            self._done_callbacks.append(fn)