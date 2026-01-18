from __future__ import annotations
from typing import TYPE_CHECKING
from .asyncio import AsyncAdapt_dbapi_connection
from .asyncio import AsyncAdapt_dbapi_cursor
from .asyncio import AsyncAdapt_dbapi_ss_cursor
from .asyncio import AsyncAdaptFallback_dbapi_connection
from .pyodbc import PyODBCConnector
from .. import pool
from .. import util
from ..util.concurrency import await_fallback
from ..util.concurrency import await_only
class aiodbcConnector(PyODBCConnector):
    is_async = True
    supports_statement_cache = True
    supports_server_side_cursors = True

    @classmethod
    def import_dbapi(cls):
        return AsyncAdapt_aioodbc_dbapi(__import__('aioodbc'), __import__('pyodbc'))

    def create_connect_args(self, url: URL) -> ConnectArgsType:
        arg, kw = super().create_connect_args(url)
        if arg and arg[0]:
            kw['dsn'] = arg[0]
        return ((), kw)

    @classmethod
    def get_pool_class(cls, url):
        async_fallback = url.query.get('async_fallback', False)
        if util.asbool(async_fallback):
            return pool.FallbackAsyncAdaptedQueuePool
        else:
            return pool.AsyncAdaptedQueuePool

    def get_driver_connection(self, connection):
        return connection._connection