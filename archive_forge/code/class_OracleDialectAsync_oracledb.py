from __future__ import annotations
import collections
import re
from typing import Any
from typing import TYPE_CHECKING
from .cx_oracle import OracleDialect_cx_oracle as _OracleDialect_cx_oracle
from ... import exc
from ... import pool
from ...connectors.asyncio import AsyncAdapt_dbapi_connection
from ...connectors.asyncio import AsyncAdapt_dbapi_cursor
from ...connectors.asyncio import AsyncAdaptFallback_dbapi_connection
from ...util import asbool
from ...util import await_fallback
from ...util import await_only
class OracleDialectAsync_oracledb(OracleDialect_oracledb):
    is_async = True
    supports_statement_cache = True
    _min_version = (2,)

    @classmethod
    def import_dbapi(cls):
        import oracledb
        return OracledbAdaptDBAPI(oracledb)

    @classmethod
    def get_pool_class(cls, url):
        async_fallback = url.query.get('async_fallback', False)
        if asbool(async_fallback):
            return pool.FallbackAsyncAdaptedQueuePool
        else:
            return pool.AsyncAdaptedQueuePool

    def get_driver_connection(self, connection):
        return connection._connection