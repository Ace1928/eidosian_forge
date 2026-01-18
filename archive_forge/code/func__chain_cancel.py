from __future__ import annotations
import warnings
from asyncio import Future
from collections import deque
from functools import partial
from itertools import chain
from typing import Any, Awaitable, Callable, NamedTuple, TypeVar, cast, overload
import zmq as _zmq
from zmq import EVENTS, POLLIN, POLLOUT
from zmq._typing import Literal
def _chain_cancel(_):
    """Chain cancellation from f to recvd"""
    if recvd.done():
        return
    if f.cancelled():
        recvd.cancel()