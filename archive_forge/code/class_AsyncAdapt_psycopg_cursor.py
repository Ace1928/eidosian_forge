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
class AsyncAdapt_psycopg_cursor:
    __slots__ = ('_cursor', 'await_', '_rows')
    _psycopg_ExecStatus = None

    def __init__(self, cursor, await_) -> None:
        self._cursor = cursor
        self.await_ = await_
        self._rows = []

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    @property
    def arraysize(self):
        return self._cursor.arraysize

    @arraysize.setter
    def arraysize(self, value):
        self._cursor.arraysize = value

    def close(self):
        self._rows.clear()
        self._cursor._close()

    def execute(self, query, params=None, **kw):
        result = self.await_(self._cursor.execute(query, params, **kw))
        res = self._cursor.pgresult
        if res and res.status == self._psycopg_ExecStatus.TUPLES_OK:
            rows = self.await_(self._cursor.fetchall())
            if not isinstance(rows, list):
                self._rows = list(rows)
            else:
                self._rows = rows
        return result

    def executemany(self, query, params_seq):
        return self.await_(self._cursor.executemany(query, params_seq))

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
            size = self._cursor.arraysize
        retval = self._rows[0:size]
        self._rows = self._rows[size:]
        return retval

    def fetchall(self):
        retval = self._rows
        self._rows = []
        return retval