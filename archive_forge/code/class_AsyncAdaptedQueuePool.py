from __future__ import annotations
import threading
import traceback
import typing
from typing import Any
from typing import cast
from typing import List
from typing import Optional
from typing import Set
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import weakref
from .base import _AsyncConnDialect
from .base import _ConnectionFairy
from .base import _ConnectionRecord
from .base import _CreatorFnType
from .base import _CreatorWRecFnType
from .base import ConnectionPoolEntry
from .base import Pool
from .base import PoolProxiedConnection
from .. import exc
from .. import util
from ..util import chop_traceback
from ..util import queue as sqla_queue
from ..util.typing import Literal
class AsyncAdaptedQueuePool(QueuePool):
    """An asyncio-compatible version of :class:`.QueuePool`.

    This pool is used by default when using :class:`.AsyncEngine` engines that
    were generated from :func:`_asyncio.create_async_engine`.   It uses an
    asyncio-compatible queue implementation that does not use
    ``threading.Lock``.

    The arguments and operation of :class:`.AsyncAdaptedQueuePool` are
    otherwise identical to that of :class:`.QueuePool`.

    """
    _is_asyncio = True
    _queue_class: Type[sqla_queue.QueueCommon[ConnectionPoolEntry]] = sqla_queue.AsyncAdaptedQueue
    _dialect = _AsyncConnDialect()