import asyncio
from functools import partial
from .base import SQLiteExecutionContext
from .pysqlite import SQLiteDialect_pysqlite
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class SQLiteExecutionContext_aiosqlite(SQLiteExecutionContext):

    def create_server_side_cursor(self):
        return self._dbapi_connection.cursor(server_side=True)