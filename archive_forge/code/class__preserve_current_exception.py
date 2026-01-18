import sys
from collections import OrderedDict
from contextlib import asynccontextmanager
from functools import partial
from ipaddress import ip_address
import itertools
import logging
import random
import ssl
import struct
import urllib.parse
from typing import List, Optional, Union
import trio
import trio.abc
from wsproto import ConnectionType, WSConnection
from wsproto.connection import ConnectionState
import wsproto.frame_protocol as wsframeproto
from wsproto.events import (
import wsproto.utilities
class _preserve_current_exception:
    """A context manager which should surround an ``__exit__`` or
    ``__aexit__`` handler or the contents of a ``finally:``
    block. It ensures that any exception that was being handled
    upon entry is not masked by a `trio.Cancelled` raised within
    the body of the context manager.

    https://github.com/python-trio/trio/issues/1559
    https://gitter.im/python-trio/general?at=5faf2293d37a1a13d6a582cf
    """
    __slots__ = ('_armed',)

    def __init__(self):
        self._armed = False

    def __enter__(self):
        self._armed = sys.exc_info()[1] is not None

    def __exit__(self, ty, value, tb):
        if value is None or not self._armed:
            return False
        if _TRIO_MULTI_ERROR:
            filtered_exception = trio.MultiError.filter(_ignore_cancel, value)
        elif isinstance(value, BaseExceptionGroup):
            filtered_exception = value.subgroup(lambda exc: not isinstance(exc, trio.Cancelled))
        else:
            filtered_exception = _ignore_cancel(value)
        return filtered_exception is None