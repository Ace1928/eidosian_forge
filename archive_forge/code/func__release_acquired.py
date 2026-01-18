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
def _release_acquired(self, key: 'ConnectionKey', proto: ResponseHandler) -> None:
    if self._closed:
        return
    try:
        self._acquired.remove(proto)
        self._drop_acquired_per_host(key, proto)
    except KeyError:
        pass
    else:
        self._release_waiter()