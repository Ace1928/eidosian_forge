from .pymysql import MySQLDialect_pymysql
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import asyncio
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class AsyncAdapt_aiomysql_dbapi:

    def __init__(self, aiomysql, pymysql):
        self.aiomysql = aiomysql
        self.pymysql = pymysql
        self.paramstyle = 'format'
        self._init_dbapi_attributes()
        self.Cursor, self.SSCursor = self._init_cursors_subclasses()

    def _init_dbapi_attributes(self):
        for name in ('Warning', 'Error', 'InterfaceError', 'DataError', 'DatabaseError', 'OperationalError', 'InterfaceError', 'IntegrityError', 'ProgrammingError', 'InternalError', 'NotSupportedError'):
            setattr(self, name, getattr(self.aiomysql, name))
        for name in ('NUMBER', 'STRING', 'DATETIME', 'BINARY', 'TIMESTAMP', 'Binary'):
            setattr(self, name, getattr(self.pymysql, name))

    def connect(self, *arg, **kw):
        async_fallback = kw.pop('async_fallback', False)
        creator_fn = kw.pop('async_creator_fn', self.aiomysql.connect)
        if util.asbool(async_fallback):
            return AsyncAdaptFallback_aiomysql_connection(self, await_fallback(creator_fn(*arg, **kw)))
        else:
            return AsyncAdapt_aiomysql_connection(self, await_only(creator_fn(*arg, **kw)))

    def _init_cursors_subclasses(self):

        class Cursor(self.aiomysql.Cursor):

            async def _show_warnings(self, conn):
                pass

        class SSCursor(self.aiomysql.SSCursor):

            async def _show_warnings(self, conn):
                pass
        return (Cursor, SSCursor)