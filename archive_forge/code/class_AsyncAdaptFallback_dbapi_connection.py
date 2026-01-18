from __future__ import annotations
import collections
import itertools
from ..engine import AdaptedConnection
from ..util.concurrency import asyncio
from ..util.concurrency import await_fallback
from ..util.concurrency import await_only
class AsyncAdaptFallback_dbapi_connection(AsyncAdapt_dbapi_connection):
    __slots__ = ()
    await_ = staticmethod(await_fallback)