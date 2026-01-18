from kazoo import client
from kazoo import exceptions as k_exc
from oslo_utils import reflection
from oslo_utils import strutils
from taskflow import exceptions as exc
from taskflow import logging
class KazooTransactionException(k_exc.KazooException):
    """Exception raised when a checked commit fails."""

    def __init__(self, message, failures):
        super(KazooTransactionException, self).__init__(message)
        self._failures = tuple(failures)

    @property
    def failures(self):
        return self._failures