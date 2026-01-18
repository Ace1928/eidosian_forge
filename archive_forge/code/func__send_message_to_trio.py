from __future__ import annotations
import contextlib
import contextvars
import inspect
import queue as stdlib_queue
import threading
from itertools import count
from typing import TYPE_CHECKING, Generic, TypeVar, overload
import attrs
import outcome
from attrs import define
from sniffio import current_async_library_cvar
import trio
from ._core import (
from ._deprecate import warn_deprecated
from ._sync import CapacityLimiter, Event
from ._util import coroutine_or_error
def _send_message_to_trio(trio_token: TrioToken | None, message_to_trio: Run[RetT] | RunSync[RetT]) -> RetT:
    """Shared logic of from_thread functions"""
    token_provided = trio_token is not None
    if not token_provided:
        try:
            trio_token = PARENT_TASK_DATA.token
        except AttributeError:
            raise RuntimeError("this thread wasn't created by Trio, pass kwarg trio_token=...") from None
    elif not isinstance(trio_token, TrioToken):
        raise RuntimeError('Passed kwarg trio_token is not of type TrioToken')
    try:
        trio.lowlevel.current_task()
    except RuntimeError:
        pass
    else:
        raise RuntimeError('this is a blocking function; call it from a thread')
    if token_provided or PARENT_TASK_DATA.abandon_on_cancel:
        message_to_trio.run_in_system_nursery(trio_token)
    else:
        message_to_trio.run_in_host_task(trio_token)
    return message_to_trio.queue.get().unwrap()