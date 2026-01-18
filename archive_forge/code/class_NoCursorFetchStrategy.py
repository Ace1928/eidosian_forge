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
class NoCursorFetchStrategy(ResultFetchStrategy):
    """Cursor strategy for a result that has no open cursor.

    There are two varieties of this strategy, one for DQL and one for
    DML (and also DDL), each of which represent a result that had a cursor
    but no longer has one.

    """
    __slots__ = ()

    def soft_close(self, result, dbapi_cursor):
        pass

    def hard_close(self, result, dbapi_cursor):
        pass

    def fetchone(self, result, dbapi_cursor, hard_close=False):
        return self._non_result(result, None)

    def fetchmany(self, result, dbapi_cursor, size=None):
        return self._non_result(result, [])

    def fetchall(self, result, dbapi_cursor):
        return self._non_result(result, [])

    def _non_result(self, result, default, err=None):
        raise NotImplementedError()