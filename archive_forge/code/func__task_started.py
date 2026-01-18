from __future__ import annotations
import array
import asyncio
import concurrent.futures
import math
import socket
import sys
import threading
from asyncio import (
from asyncio.base_events import _run_until_complete_cb  # type: ignore[attr-defined]
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Generator, Iterable
from concurrent.futures import Future
from contextlib import suppress
from contextvars import Context, copy_context
from dataclasses import dataclass
from functools import partial, wraps
from inspect import (
from io import IOBase
from os import PathLike
from queue import Queue
from signal import Signals
from socket import AddressFamily, SocketKind
from threading import Thread
from types import TracebackType
from typing import (
from weakref import WeakKeyDictionary
import sniffio
from .. import CapacityLimiterStatistics, EventStatistics, TaskInfo, abc
from .._core._eventloop import claim_worker_thread, threadlocals
from .._core._exceptions import (
from .._core._sockets import convert_ipv6_sockaddr
from .._core._streams import create_memory_object_stream
from .._core._synchronization import CapacityLimiter as BaseCapacityLimiter
from .._core._synchronization import Event as BaseEvent
from .._core._synchronization import ResourceGuard
from .._core._tasks import CancelScope as BaseCancelScope
from ..abc import (
from ..lowlevel import RunVar
from ..streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
def _task_started(task: asyncio.Task) -> bool:
    """Return ``True`` if the task has been started and has not finished."""
    try:
        return getcoroutinestate(task.get_coro()) in (CORO_RUNNING, CORO_SUSPENDED)
    except AttributeError:
        raise Exception(f'Cannot determine if task {task} has started or not') from None