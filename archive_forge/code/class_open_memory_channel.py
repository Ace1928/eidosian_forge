from __future__ import annotations
from collections import OrderedDict, deque
from math import inf
from typing import (
import attrs
from outcome import Error, Value
import trio
from ._abc import ReceiveChannel, ReceiveType, SendChannel, SendType, T
from ._core import Abort, RaiseCancelT, Task, enable_ki_protection
from ._util import NoPublicConstructor, final, generic_function
class open_memory_channel(Tuple['MemorySendChannel[T]', 'MemoryReceiveChannel[T]']):

    def __new__(cls, max_buffer_size: int | float) -> tuple[MemorySendChannel[T], MemoryReceiveChannel[T]]:
        return _open_memory_channel(max_buffer_size)

    def __init__(self, max_buffer_size: int | float):
        ...