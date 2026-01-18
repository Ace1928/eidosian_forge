from __future__ import annotations
import collections
import decimal
import json as _py_json
import re
import time
from . import json
from . import ranges
from .array import ARRAY as PGARRAY
from .base import _DECIMAL_TYPES
from .base import _FLOAT_TYPES
from .base import _INT_TYPES
from .base import ENUM
from .base import INTERVAL
from .base import OID
from .base import PGCompiler
from .base import PGDialect
from .base import PGExecutionContext
from .base import PGIdentifierPreparer
from .base import REGCLASS
from .base import REGCONFIG
from .types import BIT
from .types import BYTEA
from .types import CITEXT
from ... import exc
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...engine import processors
from ...sql import sqltypes
from ...util.concurrency import asyncio
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class AsyncAdapt_asyncpg_connection(AdaptedConnection):
    __slots__ = ('dbapi', 'isolation_level', '_isolation_setting', 'readonly', 'deferrable', '_transaction', '_started', '_prepared_statement_cache', '_prepared_statement_name_func', '_invalidate_schema_cache_asof', '_execute_mutex')
    await_ = staticmethod(await_only)

    def __init__(self, dbapi, connection, prepared_statement_cache_size=100, prepared_statement_name_func=None):
        self.dbapi = dbapi
        self._connection = connection
        self.isolation_level = self._isolation_setting = 'read_committed'
        self.readonly = False
        self.deferrable = False
        self._transaction = None
        self._started = False
        self._invalidate_schema_cache_asof = time.time()
        self._execute_mutex = asyncio.Lock()
        if prepared_statement_cache_size:
            self._prepared_statement_cache = util.LRUCache(prepared_statement_cache_size)
        else:
            self._prepared_statement_cache = None
        if prepared_statement_name_func:
            self._prepared_statement_name_func = prepared_statement_name_func
        else:
            self._prepared_statement_name_func = self._default_name_func

    async def _check_type_cache_invalidation(self, invalidate_timestamp):
        if invalidate_timestamp > self._invalidate_schema_cache_asof:
            await self._connection.reload_schema_state()
            self._invalidate_schema_cache_asof = invalidate_timestamp

    async def _prepare(self, operation, invalidate_timestamp):
        await self._check_type_cache_invalidation(invalidate_timestamp)
        cache = self._prepared_statement_cache
        if cache is None:
            prepared_stmt = await self._connection.prepare(operation, name=self._prepared_statement_name_func())
            attributes = prepared_stmt.get_attributes()
            return (prepared_stmt, attributes)
        if operation in cache:
            prepared_stmt, attributes, cached_timestamp = cache[operation]
            if cached_timestamp > invalidate_timestamp:
                return (prepared_stmt, attributes)
        prepared_stmt = await self._connection.prepare(operation, name=self._prepared_statement_name_func())
        attributes = prepared_stmt.get_attributes()
        cache[operation] = (prepared_stmt, attributes, time.time())
        return (prepared_stmt, attributes)

    def _handle_exception(self, error):
        if self._connection.is_closed():
            self._transaction = None
            self._started = False
        if not isinstance(error, AsyncAdapt_asyncpg_dbapi.Error):
            exception_mapping = self.dbapi._asyncpg_error_translate
            for super_ in type(error).__mro__:
                if super_ in exception_mapping:
                    translated_error = exception_mapping[super_]('%s: %s' % (type(error), error))
                    translated_error.pgcode = translated_error.sqlstate = getattr(error, 'sqlstate', None)
                    raise translated_error from error
            else:
                raise error
        else:
            raise error

    @property
    def autocommit(self):
        return self.isolation_level == 'autocommit'

    @autocommit.setter
    def autocommit(self, value):
        if value:
            self.isolation_level = 'autocommit'
        else:
            self.isolation_level = self._isolation_setting

    def ping(self):
        try:
            _ = self.await_(self._async_ping())
        except Exception as error:
            self._handle_exception(error)

    async def _async_ping(self):
        if self._transaction is None and self.isolation_level != 'autocommit':
            tr = self._connection.transaction()
            await tr.start()
            try:
                await self._connection.fetchrow(';')
            finally:
                await tr.rollback()
        else:
            await self._connection.fetchrow(';')

    def set_isolation_level(self, level):
        if self._started:
            self.rollback()
        self.isolation_level = self._isolation_setting = level

    async def _start_transaction(self):
        if self.isolation_level == 'autocommit':
            return
        try:
            self._transaction = self._connection.transaction(isolation=self.isolation_level, readonly=self.readonly, deferrable=self.deferrable)
            await self._transaction.start()
        except Exception as error:
            self._handle_exception(error)
        else:
            self._started = True

    def cursor(self, server_side=False):
        if server_side:
            return AsyncAdapt_asyncpg_ss_cursor(self)
        else:
            return AsyncAdapt_asyncpg_cursor(self)

    def rollback(self):
        if self._started:
            try:
                self.await_(self._transaction.rollback())
            except Exception as error:
                self._handle_exception(error)
            finally:
                self._transaction = None
                self._started = False

    def commit(self):
        if self._started:
            try:
                self.await_(self._transaction.commit())
            except Exception as error:
                self._handle_exception(error)
            finally:
                self._transaction = None
                self._started = False

    def close(self):
        self.rollback()
        self.await_(self._connection.close())

    def terminate(self):
        if util.concurrency.in_greenlet():
            try:
                self.await_(self._connection.close(timeout=2))
            except (asyncio.TimeoutError, OSError, self.dbapi.asyncpg.PostgresError):
                self._connection.terminate()
        else:
            self._connection.terminate()
        self._started = False

    @staticmethod
    def _default_name_func():
        return None