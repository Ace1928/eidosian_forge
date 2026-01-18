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
class AsyncAdapt_aioodbc_dbapi:

    def __init__(self, aioodbc, pyodbc):
        self.aioodbc = aioodbc
        self.pyodbc = pyodbc
        self.paramstyle = pyodbc.paramstyle
        self._init_dbapi_attributes()
        self.Cursor = AsyncAdapt_dbapi_cursor
        self.version = pyodbc.version

    def _init_dbapi_attributes(self):
        for name in ('Warning', 'Error', 'InterfaceError', 'DataError', 'DatabaseError', 'OperationalError', 'InterfaceError', 'IntegrityError', 'ProgrammingError', 'InternalError', 'NotSupportedError', 'NUMBER', 'STRING', 'DATETIME', 'BINARY', 'Binary', 'BinaryNull', 'SQL_VARCHAR', 'SQL_WVARCHAR'):
            setattr(self, name, getattr(self.pyodbc, name))

    def connect(self, *arg, **kw):
        async_fallback = kw.pop('async_fallback', False)
        creator_fn = kw.pop('async_creator_fn', self.aioodbc.connect)
        if util.asbool(async_fallback):
            return AsyncAdaptFallback_aioodbc_connection(self, await_fallback(creator_fn(*arg, **kw)))
        else:
            return AsyncAdapt_aioodbc_connection(self, await_only(creator_fn(*arg, **kw)))