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
def _available_connections(self, key: 'ConnectionKey') -> int:
    """
        Return number of available connections.

        The limit, limit_per_host and the connection key are taken into account.

        If it returns less than 1 means that there are no connections
        available.
        """
    if self._limit:
        available = self._limit - len(self._acquired)
        if self._limit_per_host and available > 0 and (key in self._acquired_per_host):
            acquired = self._acquired_per_host.get(key)
            assert acquired is not None
            available = self._limit_per_host - len(acquired)
    elif self._limit_per_host and key in self._acquired_per_host:
        acquired = self._acquired_per_host.get(key)
        assert acquired is not None
        available = self._limit_per_host - len(acquired)
    else:
        available = 1
    return available