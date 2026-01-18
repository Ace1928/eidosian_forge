from contextlib import asynccontextmanager
from .pymysql import MySQLDialect_pymysql
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import asyncio
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only
class AsyncAdapt_asyncmy_connection(AdaptedConnection):
    await_ = staticmethod(await_only)
    __slots__ = ('dbapi', '_execute_mutex')

    def __init__(self, dbapi, connection):
        self.dbapi = dbapi
        self._connection = connection
        self._execute_mutex = asyncio.Lock()

    @asynccontextmanager
    async def _mutex_and_adapt_errors(self):
        async with self._execute_mutex:
            try:
                yield
            except AttributeError:
                raise self.dbapi.InternalError('network operation failed due to asyncmy attribute error')

    def ping(self, reconnect):
        assert not reconnect
        return self.await_(self._do_ping())

    async def _do_ping(self):
        async with self._mutex_and_adapt_errors():
            return await self._connection.ping(False)

    def character_set_name(self):
        return self._connection.character_set_name()

    def autocommit(self, value):
        self.await_(self._connection.autocommit(value))

    def cursor(self, server_side=False):
        if server_side:
            return AsyncAdapt_asyncmy_ss_cursor(self)
        else:
            return AsyncAdapt_asyncmy_cursor(self)

    def rollback(self):
        self.await_(self._connection.rollback())

    def commit(self):
        self.await_(self._connection.commit())

    def terminate(self):
        self._connection.close()

    def close(self) -> None:
        self.await_(self._connection.ensure_closed())