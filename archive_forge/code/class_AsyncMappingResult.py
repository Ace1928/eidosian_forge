from __future__ import annotations
import operator
from typing import Any
from typing import AsyncIterator
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from . import exc as async_exc
from ... import util
from ...engine import Result
from ...engine.result import _NO_ROW
from ...engine.result import _R
from ...engine.result import _WithKeys
from ...engine.result import FilterResult
from ...engine.result import FrozenResult
from ...engine.result import ResultMetaData
from ...engine.row import Row
from ...engine.row import RowMapping
from ...sql.base import _generative
from ...util.concurrency import greenlet_spawn
from ...util.typing import Literal
from ...util.typing import Self
class AsyncMappingResult(_WithKeys, AsyncCommon[RowMapping]):
    """A wrapper for a :class:`_asyncio.AsyncResult` that returns dictionary
    values rather than :class:`_engine.Row` values.

    The :class:`_asyncio.AsyncMappingResult` object is acquired by calling the
    :meth:`_asyncio.AsyncResult.mappings` method.

    Refer to the :class:`_result.MappingResult` object in the synchronous
    SQLAlchemy API for a complete behavioral description.

    .. versionadded:: 1.4

    """
    __slots__ = ()
    _generate_rows = True
    _post_creational_filter = operator.attrgetter('_mapping')

    def __init__(self, result: Result[Any]):
        self._real_result = result
        self._unique_filter_state = result._unique_filter_state
        self._metadata = result._metadata
        if result._source_supports_scalars:
            self._metadata = self._metadata._reduce([0])

    def unique(self, strategy: Optional[_UniqueFilterType]=None) -> Self:
        """Apply unique filtering to the objects returned by this
        :class:`_asyncio.AsyncMappingResult`.

        See :meth:`_asyncio.AsyncResult.unique` for usage details.

        """
        self._unique_filter_state = (set(), strategy)
        return self

    def columns(self, *col_expressions: _KeyIndexType) -> Self:
        """Establish the columns that should be returned in each row."""
        return self._column_slices(col_expressions)

    async def partitions(self, size: Optional[int]=None) -> AsyncIterator[Sequence[RowMapping]]:
        """Iterate through sub-lists of elements of the size given.

        Equivalent to :meth:`_asyncio.AsyncResult.partitions` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.

        """
        getter = self._manyrow_getter
        while True:
            partition = await greenlet_spawn(getter, self, size)
            if partition:
                yield partition
            else:
                break

    async def fetchall(self) -> Sequence[RowMapping]:
        """A synonym for the :meth:`_asyncio.AsyncMappingResult.all` method."""
        return await greenlet_spawn(self._allrows)

    async def fetchone(self) -> Optional[RowMapping]:
        """Fetch one object.

        Equivalent to :meth:`_asyncio.AsyncResult.fetchone` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.

        """
        row = await greenlet_spawn(self._onerow_getter, self)
        if row is _NO_ROW:
            return None
        else:
            return row

    async def fetchmany(self, size: Optional[int]=None) -> Sequence[RowMapping]:
        """Fetch many rows.

        Equivalent to :meth:`_asyncio.AsyncResult.fetchmany` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.

        """
        return await greenlet_spawn(self._manyrow_getter, self, size)

    async def all(self) -> Sequence[RowMapping]:
        """Return all rows in a list.

        Equivalent to :meth:`_asyncio.AsyncResult.all` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.

        """
        return await greenlet_spawn(self._allrows)

    def __aiter__(self) -> AsyncMappingResult:
        return self

    async def __anext__(self) -> RowMapping:
        row = await greenlet_spawn(self._onerow_getter, self)
        if row is _NO_ROW:
            raise StopAsyncIteration()
        else:
            return row

    async def first(self) -> Optional[RowMapping]:
        """Fetch the first object or ``None`` if no object is present.

        Equivalent to :meth:`_asyncio.AsyncResult.first` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.

        """
        return await greenlet_spawn(self._only_one_row, False, False, False)

    async def one_or_none(self) -> Optional[RowMapping]:
        """Return at most one object or raise an exception.

        Equivalent to :meth:`_asyncio.AsyncResult.one_or_none` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.

        """
        return await greenlet_spawn(self._only_one_row, True, False, False)

    async def one(self) -> RowMapping:
        """Return exactly one object or raise an exception.

        Equivalent to :meth:`_asyncio.AsyncResult.one` except that
        :class:`_engine.RowMapping` values, rather than :class:`_engine.Row`
        objects, are returned.

        """
        return await greenlet_spawn(self._only_one_row, True, True, False)