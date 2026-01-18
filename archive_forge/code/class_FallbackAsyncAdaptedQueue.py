from __future__ import annotations
import asyncio
from collections import deque
import threading
from time import time as _time
import typing
from typing import Any
from typing import Awaitable
from typing import Deque
from typing import Generic
from typing import Optional
from typing import TypeVar
from .concurrency import await_fallback
from .concurrency import await_only
from .langhelpers import memoized_property
class FallbackAsyncAdaptedQueue(AsyncAdaptedQueue[_T]):
    if not typing.TYPE_CHECKING:
        await_ = staticmethod(await_fallback)