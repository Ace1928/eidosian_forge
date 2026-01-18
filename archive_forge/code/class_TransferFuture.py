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
class TransferFuture(BaseTransferFuture):

    def __init__(self, meta=None, coordinator=None):
        """The future associated to a submitted transfer request

        :type meta: TransferMeta
        :param meta: The metadata associated to the request. This object
            is visible to the requester.

        :type coordinator: TransferCoordinator
        :param coordinator: The coordinator associated to the request. This
            object is not visible to the requester.
        """
        self._meta = meta
        if meta is None:
            self._meta = TransferMeta()
        self._coordinator = coordinator
        if coordinator is None:
            self._coordinator = TransferCoordinator()

    @property
    def meta(self):
        return self._meta

    def done(self):
        return self._coordinator.done()

    def result(self):
        try:
            return self._coordinator.result()
        except KeyboardInterrupt as e:
            self.cancel()
            raise e

    def cancel(self):
        self._coordinator.cancel()

    def set_exception(self, exception):
        """Sets the exception on the future."""
        if not self.done():
            raise TransferNotDoneError('set_exception can only be called once the transfer is complete.')
        self._coordinator.set_exception(exception, override=True)