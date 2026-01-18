import asyncio
import functools
import random
import sys
import traceback
import warnings
from collections import defaultdict, deque
from contextlib import suppress
from http import HTTPStatus
from http.cookies import SimpleCookie
from itertools import cycle, islice
from time import monotonic
from types import TracebackType
from typing import (
import attr
from . import hdrs, helpers
from .abc import AbstractResolver
from .client_exceptions import (
from .client_proto import ResponseHandler
from .client_reqrep import ClientRequest, Fingerprint, _merge_ssl_params
from .helpers import ceil_timeout, get_running_loop, is_ip_address, noop, sentinel
from .locks import EventResultOrError
from .resolver import DefaultResolver
def _cleanup_closed(self) -> None:
    """Double confirmation for transport close.

        Some broken ssl servers may leave socket open without proper close.
        """
    if self._cleanup_closed_handle:
        self._cleanup_closed_handle.cancel()
    for transport in self._cleanup_closed_transports:
        if transport is not None:
            transport.abort()
    self._cleanup_closed_transports = []
    if not self._cleanup_closed_disabled:
        self._cleanup_closed_handle = helpers.weakref_handle(self, '_cleanup_closed', self._cleanup_closed_period, self._loop, timeout_ceil_threshold=self._timeout_ceil_threshold)