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
class AsyncAdapt_psycopg_ss_cursor(AsyncAdapt_psycopg_cursor):

    def execute(self, query, params=None, **kw):
        self.await_(self._cursor.execute(query, params, **kw))
        return self

    def close(self):
        self.await_(self._cursor.close())

    def fetchone(self):
        return self.await_(self._cursor.fetchone())

    def fetchmany(self, size=0):
        return self.await_(self._cursor.fetchmany(size))

    def fetchall(self):
        return self.await_(self._cursor.fetchall())

    def __iter__(self):
        iterator = self._cursor.__aiter__()
        while True:
            try:
                yield self.await_(iterator.__anext__())
            except StopAsyncIteration:
                break