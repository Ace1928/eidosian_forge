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
class AsyncAdapt_asyncpg_cursor:
    __slots__ = ('_adapt_connection', '_connection', '_rows', 'description', 'arraysize', 'rowcount', '_cursor', '_invalidate_schema_cache_asof')
    server_side = False

    def __init__(self, adapt_connection):
        self._adapt_connection = adapt_connection
        self._connection = adapt_connection._connection
        self._rows = []
        self._cursor = None
        self.description = None
        self.arraysize = 1
        self.rowcount = -1
        self._invalidate_schema_cache_asof = 0

    def close(self):
        self._rows[:] = []

    def _handle_exception(self, error):
        self._adapt_connection._handle_exception(error)

    async def _prepare_and_execute(self, operation, parameters):
        adapt_connection = self._adapt_connection
        async with adapt_connection._execute_mutex:
            if not adapt_connection._started:
                await adapt_connection._start_transaction()
            if parameters is None:
                parameters = ()
            try:
                prepared_stmt, attributes = await adapt_connection._prepare(operation, self._invalidate_schema_cache_asof)
                if attributes:
                    self.description = [(attr.name, attr.type.oid, None, None, None, None, None) for attr in attributes]
                else:
                    self.description = None
                if self.server_side:
                    self._cursor = await prepared_stmt.cursor(*parameters)
                    self.rowcount = -1
                else:
                    self._rows = await prepared_stmt.fetch(*parameters)
                    status = prepared_stmt.get_statusmsg()
                    reg = re.match('(?:SELECT|UPDATE|DELETE|INSERT \\d+) (\\d+)', status)
                    if reg:
                        self.rowcount = int(reg.group(1))
                    else:
                        self.rowcount = -1
            except Exception as error:
                self._handle_exception(error)

    async def _executemany(self, operation, seq_of_parameters):
        adapt_connection = self._adapt_connection
        self.description = None
        async with adapt_connection._execute_mutex:
            await adapt_connection._check_type_cache_invalidation(self._invalidate_schema_cache_asof)
            if not adapt_connection._started:
                await adapt_connection._start_transaction()
            try:
                return await self._connection.executemany(operation, seq_of_parameters)
            except Exception as error:
                self._handle_exception(error)

    def execute(self, operation, parameters=None):
        self._adapt_connection.await_(self._prepare_and_execute(operation, parameters))

    def executemany(self, operation, seq_of_parameters):
        return self._adapt_connection.await_(self._executemany(operation, seq_of_parameters))

    def setinputsizes(self, *inputsizes):
        raise NotImplementedError()

    def __iter__(self):
        while self._rows:
            yield self._rows.pop(0)

    def fetchone(self):
        if self._rows:
            return self._rows.pop(0)
        else:
            return None

    def fetchmany(self, size=None):
        if size is None:
            size = self.arraysize
        retval = self._rows[0:size]
        self._rows[:] = self._rows[size:]
        return retval

    def fetchall(self):
        retval = self._rows[:]
        self._rows[:] = []
        return retval