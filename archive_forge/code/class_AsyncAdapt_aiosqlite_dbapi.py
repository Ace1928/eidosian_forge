import asyncio
from functools import partial
from .base import SQLiteExecutionContext
from .pysqlite import SQLiteDialect_pysqlite
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class AsyncAdapt_aiosqlite_dbapi:

    def __init__(self, aiosqlite, sqlite):
        self.aiosqlite = aiosqlite
        self.sqlite = sqlite
        self.paramstyle = 'qmark'
        self._init_dbapi_attributes()

    def _init_dbapi_attributes(self):
        for name in ('DatabaseError', 'Error', 'IntegrityError', 'NotSupportedError', 'OperationalError', 'ProgrammingError', 'sqlite_version', 'sqlite_version_info'):
            setattr(self, name, getattr(self.aiosqlite, name))
        for name in ('PARSE_COLNAMES', 'PARSE_DECLTYPES'):
            setattr(self, name, getattr(self.sqlite, name))
        for name in ('Binary',):
            setattr(self, name, getattr(self.sqlite, name))

    def connect(self, *arg, **kw):
        async_fallback = kw.pop('async_fallback', False)
        creator_fn = kw.pop('async_creator_fn', None)
        if creator_fn:
            connection = creator_fn(*arg, **kw)
        else:
            connection = self.aiosqlite.connect(*arg, **kw)
            connection.daemon = True
        if util.asbool(async_fallback):
            return AsyncAdaptFallback_aiosqlite_connection(self, await_fallback(connection))
        else:
            return AsyncAdapt_aiosqlite_connection(self, await_only(connection))