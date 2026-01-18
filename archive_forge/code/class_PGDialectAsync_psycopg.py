from __future__ import annotations
import logging
import re
from typing import cast
from typing import TYPE_CHECKING
from . import ranges
from ._psycopg_common import _PGDialect_common_psycopg
from ._psycopg_common import _PGExecutionContext_common_psycopg
from .base import INTERVAL
from .base import PGCompiler
from .base import PGIdentifierPreparer
from .base import REGCONFIG
from .json import JSON
from .json import JSONB
from .json import JSONPathType
from .types import CITEXT
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...sql import sqltypes
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class PGDialectAsync_psycopg(PGDialect_psycopg):
    is_async = True
    supports_statement_cache = True

    @classmethod
    def import_dbapi(cls):
        import psycopg
        from psycopg.pq import ExecStatus
        AsyncAdapt_psycopg_cursor._psycopg_ExecStatus = ExecStatus
        return PsycopgAdaptDBAPI(psycopg)

    @classmethod
    def get_pool_class(cls, url):
        async_fallback = url.query.get('async_fallback', False)
        if util.asbool(async_fallback):
            return pool.FallbackAsyncAdaptedQueuePool
        else:
            return pool.AsyncAdaptedQueuePool

    def _type_info_fetch(self, connection, name):
        from psycopg.types import TypeInfo
        adapted = connection.connection
        return adapted.await_(TypeInfo.fetch(adapted.driver_connection, name))

    def _do_isolation_level(self, connection, autocommit, isolation_level):
        connection.set_autocommit(autocommit)
        connection.set_isolation_level(isolation_level)

    def _do_autocommit(self, connection, value):
        connection.set_autocommit(value)

    def set_readonly(self, connection, value):
        connection.set_read_only(value)

    def set_deferrable(self, connection, value):
        connection.set_deferrable(value)

    def get_driver_connection(self, connection):
        return connection._connection