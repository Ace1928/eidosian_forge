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
class FullyBufferedCursorFetchStrategy(CursorFetchStrategy):
    """A cursor strategy that buffers rows fully upon creation.

    Used for operations where a result is to be delivered
    after the database conversation can not be continued,
    such as MSSQL INSERT...OUTPUT after an autocommit.

    """
    __slots__ = ('_rowbuffer', 'alternate_cursor_description')

    def __init__(self, dbapi_cursor, alternate_description=None, initial_buffer=None):
        self.alternate_cursor_description = alternate_description
        if initial_buffer is not None:
            self._rowbuffer = collections.deque(initial_buffer)
        else:
            self._rowbuffer = collections.deque(dbapi_cursor.fetchall())

    def yield_per(self, result, dbapi_cursor, num):
        pass

    def soft_close(self, result, dbapi_cursor):
        self._rowbuffer.clear()
        super().soft_close(result, dbapi_cursor)

    def hard_close(self, result, dbapi_cursor):
        self._rowbuffer.clear()
        super().hard_close(result, dbapi_cursor)

    def fetchone(self, result, dbapi_cursor, hard_close=False):
        if self._rowbuffer:
            return self._rowbuffer.popleft()
        else:
            result._soft_close(hard=hard_close)
            return None

    def fetchmany(self, result, dbapi_cursor, size=None):
        if size is None:
            return self.fetchall(result, dbapi_cursor)
        buf = list(self._rowbuffer)
        rows = buf[0:size]
        self._rowbuffer = collections.deque(buf[size:])
        if not rows:
            result._soft_close()
        return rows

    def fetchall(self, result, dbapi_cursor):
        ret = self._rowbuffer
        self._rowbuffer = collections.deque()
        result._soft_close()
        return ret