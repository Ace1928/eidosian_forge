from __future__ import annotations
import collections
import itertools
from ..engine import AdaptedConnection
from ..util.concurrency import asyncio
from ..util.concurrency import await_fallback
from ..util.concurrency import await_only
class AsyncAdapt_dbapi_connection(AdaptedConnection):
    _cursor_cls = AsyncAdapt_dbapi_cursor
    _ss_cursor_cls = AsyncAdapt_dbapi_ss_cursor
    await_ = staticmethod(await_only)
    __slots__ = ('dbapi', '_execute_mutex')

    def __init__(self, dbapi, connection):
        self.dbapi = dbapi
        self._connection = connection
        self._execute_mutex = asyncio.Lock()

    def ping(self, reconnect):
        return self.await_(self._connection.ping(reconnect))

    def add_output_converter(self, *arg, **kw):
        self._connection.add_output_converter(*arg, **kw)

    def character_set_name(self):
        return self._connection.character_set_name()

    @property
    def autocommit(self):
        return self._connection.autocommit

    @autocommit.setter
    def autocommit(self, value):
        self._connection._conn.autocommit = value

    def cursor(self, server_side=False):
        if server_side:
            return self._ss_cursor_cls(self)
        else:
            return self._cursor_cls(self)

    def rollback(self):
        self.await_(self._connection.rollback())

    def commit(self):
        self.await_(self._connection.commit())

    def close(self):
        self.await_(self._connection.close())