from __future__ import annotations
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from types import TracebackType
from typing import Generic, NamedTuple, TypeVar
from .. import (
from ..abc import Event, ObjectReceiveStream, ObjectSendStream
from ..lowlevel import checkpoint
class MemoryObjectStreamStatistics(NamedTuple):
    current_buffer_used: int
    max_buffer_size: float
    open_send_streams: int
    open_receive_streams: int
    tasks_waiting_send: int
    tasks_waiting_receive: int