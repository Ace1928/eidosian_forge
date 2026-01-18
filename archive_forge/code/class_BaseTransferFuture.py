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
class BaseTransferFuture(object):

    @property
    def meta(self):
        """The metadata associated to the TransferFuture"""
        raise NotImplementedError('meta')

    def done(self):
        """Determines if a TransferFuture has completed

        :returns: True if completed. False, otherwise.
        """
        raise NotImplementedError('done()')

    def result(self):
        """Waits until TransferFuture is done and returns the result

        If the TransferFuture succeeded, it will return the result. If the
        TransferFuture failed, it will raise the exception associated to the
        failure.
        """
        raise NotImplementedError('result()')

    def cancel(self):
        """Cancels the request associated with the TransferFuture"""
        raise NotImplementedError('cancel()')