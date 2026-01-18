import asyncio
from functools import partial
from .base import SQLiteExecutionContext
from .pysqlite import SQLiteDialect_pysqlite
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class SQLiteDialect_aiosqlite(SQLiteDialect_pysqlite):
    driver = 'aiosqlite'
    supports_statement_cache = True
    is_async = True
    supports_server_side_cursors = True
    execution_ctx_cls = SQLiteExecutionContext_aiosqlite

    @classmethod
    def import_dbapi(cls):
        return AsyncAdapt_aiosqlite_dbapi(__import__('aiosqlite'), __import__('sqlite3'))

    @classmethod
    def get_pool_class(cls, url):
        if cls._is_url_file_db(url):
            return pool.NullPool
        else:
            return pool.StaticPool

    def is_disconnect(self, e, connection, cursor):
        if isinstance(e, self.dbapi.OperationalError) and 'no active connection' in str(e):
            return True
        return super().is_disconnect(e, connection, cursor)

    def get_driver_connection(self, connection):
        return connection._connection