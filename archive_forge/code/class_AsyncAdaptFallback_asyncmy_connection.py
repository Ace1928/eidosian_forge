from contextlib import asynccontextmanager
from .pymysql import MySQLDialect_pymysql
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import asyncio
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class AsyncAdaptFallback_asyncmy_connection(AsyncAdapt_asyncmy_connection):
    __slots__ = ()
    await_ = staticmethod(await_fallback)