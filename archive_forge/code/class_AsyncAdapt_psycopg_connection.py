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
class AsyncAdapt_psycopg_connection(AdaptedConnection):
    _connection: AsyncConnection
    __slots__ = ()
    await_ = staticmethod(await_only)

    def __init__(self, connection) -> None:
        self._connection = connection

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def execute(self, query, params=None, **kw):
        cursor = self.await_(self._connection.execute(query, params, **kw))
        return AsyncAdapt_psycopg_cursor(cursor, self.await_)

    def cursor(self, *args, **kw):
        cursor = self._connection.cursor(*args, **kw)
        if hasattr(cursor, 'name'):
            return AsyncAdapt_psycopg_ss_cursor(cursor, self.await_)
        else:
            return AsyncAdapt_psycopg_cursor(cursor, self.await_)

    def commit(self):
        self.await_(self._connection.commit())

    def rollback(self):
        self.await_(self._connection.rollback())

    def close(self):
        self.await_(self._connection.close())

    @property
    def autocommit(self):
        return self._connection.autocommit

    @autocommit.setter
    def autocommit(self, value):
        self.set_autocommit(value)

    def set_autocommit(self, value):
        self.await_(self._connection.set_autocommit(value))

    def set_isolation_level(self, value):
        self.await_(self._connection.set_isolation_level(value))

    def set_read_only(self, value):
        self.await_(self._connection.set_read_only(value))

    def set_deferrable(self, value):
        self.await_(self._connection.set_deferrable(value))