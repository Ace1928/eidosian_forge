from .pymysql import MySQLDialect_pymysql
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import asyncio
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class AsyncAdaptFallback_aiomysql_connection(AsyncAdapt_aiomysql_connection):
    __slots__ = ()
    await_ = staticmethod(await_fallback)