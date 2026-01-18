from __future__ import annotations
import collections
import functools
import operator
import typing
from typing import Any
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .result import IteratorResult
from .result import MergedResult
from .result import Result
from .result import ResultMetaData
from .result import SimpleResultMetaData
from .result import tuplegetter
from .row import Row
from .. import exc
from .. import util
from ..sql import elements
from ..sql import sqltypes
from ..sql import util as sql_util
from ..sql.base import _generative
from ..sql.compiler import ResultColumnsEntry
from ..sql.compiler import RM_NAME
from ..sql.compiler import RM_OBJECTS
from ..sql.compiler import RM_RENDERED_NAME
from ..sql.compiler import RM_TYPE
from ..sql.type_api import TypeEngine
from ..util import compat
from ..util.typing import Literal
from ..util.typing import Self
class BufferedRowCursorFetchStrategy(CursorFetchStrategy):
    """A cursor fetch strategy with row buffering behavior.

    This strategy buffers the contents of a selection of rows
    before ``fetchone()`` is called.  This is to allow the results of
    ``cursor.description`` to be available immediately, when
    interfacing with a DB-API that requires rows to be consumed before
    this information is available (currently psycopg2, when used with
    server-side cursors).

    The pre-fetching behavior fetches only one row initially, and then
    grows its buffer size by a fixed amount with each successive need
    for additional rows up the ``max_row_buffer`` size, which defaults
    to 1000::

        with psycopg2_engine.connect() as conn:

            result = conn.execution_options(
                stream_results=True, max_row_buffer=50
                ).execute(text("select * from table"))

    .. versionadded:: 1.4 ``max_row_buffer`` may now exceed 1000 rows.

    .. seealso::

        :ref:`psycopg2_execution_options`
    """
    __slots__ = ('_max_row_buffer', '_rowbuffer', '_bufsize', '_growth_factor')

    def __init__(self, dbapi_cursor, execution_options, growth_factor=5, initial_buffer=None):
        self._max_row_buffer = execution_options.get('max_row_buffer', 1000)
        if initial_buffer is not None:
            self._rowbuffer = initial_buffer
        else:
            self._rowbuffer = collections.deque(dbapi_cursor.fetchmany(1))
        self._growth_factor = growth_factor
        if growth_factor:
            self._bufsize = min(self._max_row_buffer, self._growth_factor)
        else:
            self._bufsize = self._max_row_buffer

    @classmethod
    def create(cls, result):
        return BufferedRowCursorFetchStrategy(result.cursor, result.context.execution_options)

    def _buffer_rows(self, result, dbapi_cursor):
        """this is currently used only by fetchone()."""
        size = self._bufsize
        try:
            if size < 1:
                new_rows = dbapi_cursor.fetchall()
            else:
                new_rows = dbapi_cursor.fetchmany(size)
        except BaseException as e:
            self.handle_exception(result, dbapi_cursor, e)
        if not new_rows:
            return
        self._rowbuffer = collections.deque(new_rows)
        if self._growth_factor and size < self._max_row_buffer:
            self._bufsize = min(self._max_row_buffer, size * self._growth_factor)

    def yield_per(self, result, dbapi_cursor, num):
        self._growth_factor = 0
        self._max_row_buffer = self._bufsize = num

    def soft_close(self, result, dbapi_cursor):
        self._rowbuffer.clear()
        super().soft_close(result, dbapi_cursor)

    def hard_close(self, result, dbapi_cursor):
        self._rowbuffer.clear()
        super().hard_close(result, dbapi_cursor)

    def fetchone(self, result, dbapi_cursor, hard_close=False):
        if not self._rowbuffer:
            self._buffer_rows(result, dbapi_cursor)
            if not self._rowbuffer:
                try:
                    result._soft_close(hard=hard_close)
                except BaseException as e:
                    self.handle_exception(result, dbapi_cursor, e)
                return None
        return self._rowbuffer.popleft()

    def fetchmany(self, result, dbapi_cursor, size=None):
        if size is None:
            return self.fetchall(result, dbapi_cursor)
        buf = list(self._rowbuffer)
        lb = len(buf)
        if size > lb:
            try:
                new = dbapi_cursor.fetchmany(size - lb)
            except BaseException as e:
                self.handle_exception(result, dbapi_cursor, e)
            else:
                if not new:
                    result._soft_close()
                else:
                    buf.extend(new)
        result = buf[0:size]
        self._rowbuffer = collections.deque(buf[size:])
        return result

    def fetchall(self, result, dbapi_cursor):
        try:
            ret = list(self._rowbuffer) + list(dbapi_cursor.fetchall())
            self._rowbuffer.clear()
            result._soft_close()
            return ret
        except BaseException as e:
            self.handle_exception(result, dbapi_cursor, e)