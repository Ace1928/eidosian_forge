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
class AsyncAdapt_oracledb_cursor(AsyncAdapt_dbapi_cursor):
    _cursor: AsyncCursor
    __slots__ = ()

    @property
    def outputtypehandler(self):
        return self._cursor.outputtypehandler

    @outputtypehandler.setter
    def outputtypehandler(self, value):
        self._cursor.outputtypehandler = value

    def var(self, *args, **kwargs):
        return self._cursor.var(*args, **kwargs)

    def close(self):
        self._rows.clear()
        self._cursor.close()

    def setinputsizes(self, *args: Any, **kwargs: Any) -> Any:
        return self._cursor.setinputsizes(*args, **kwargs)

    def _aenter_cursor(self, cursor: AsyncCursor) -> AsyncCursor:
        try:
            return cursor.__enter__()
        except Exception as error:
            self._adapt_connection._handle_exception(error)

    async def _execute_async(self, operation, parameters):
        if parameters is None:
            result = await self._cursor.execute(operation)
        else:
            result = await self._cursor.execute(operation, parameters)
        if self._cursor.description and (not self.server_side):
            self._rows = collections.deque(await self._cursor.fetchall())
        return result

    async def _executemany_async(self, operation, seq_of_parameters):
        return await self._cursor.executemany(operation, seq_of_parameters)

    def __enter__(self):
        return self

    def __exit__(self, type_: Any, value: Any, traceback: Any) -> None:
        self.close()