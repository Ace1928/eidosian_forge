from __future__ import absolute_import
import functools
import logging
import numbers
import os
import time
import requests.adapters  # pylint: disable=ungrouped-imports
import requests.exceptions  # pylint: disable=ungrouped-imports
from requests.packages.urllib3.util.ssl_ import (  # type: ignore
import six  # pylint: disable=ungrouped-imports
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
import google.auth.transport._mtls_helper
from google.oauth2 import service_account
class TimeoutGuard(object):
    """A context manager raising an error if the suite execution took too long.

    Args:
        timeout (Union[None, Union[float, Tuple[float, float]]]):
            The maximum number of seconds a suite can run without the context
            manager raising a timeout exception on exit. If passed as a tuple,
            the smaller of the values is taken as a timeout. If ``None``, a
            timeout error is never raised.
        timeout_error_type (Optional[Exception]):
            The type of the error to raise on timeout. Defaults to
            :class:`requests.exceptions.Timeout`.
    """

    def __init__(self, timeout, timeout_error_type=requests.exceptions.Timeout):
        self._timeout = timeout
        self.remaining_timeout = timeout
        self._timeout_error_type = timeout_error_type

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value:
            return
        if self._timeout is None:
            return
        elapsed = time.time() - self._start
        deadline_hit = False
        if isinstance(self._timeout, numbers.Number):
            self.remaining_timeout = self._timeout - elapsed
            deadline_hit = self.remaining_timeout <= 0
        else:
            self.remaining_timeout = tuple((x - elapsed for x in self._timeout))
            deadline_hit = min(self.remaining_timeout) <= 0
        if deadline_hit:
            raise self._timeout_error_type()