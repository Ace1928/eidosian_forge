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
class AsyncScalarResult(AsyncCommon[_R]):
    """A wrapper for a :class:`_asyncio.AsyncResult` that returns scalar values
    rather than :class:`_row.Row` values.

    The :class:`_asyncio.AsyncScalarResult` object is acquired by calling the
    :meth:`_asyncio.AsyncResult.scalars` method.

    Refer to the :class:`_result.ScalarResult` object in the synchronous
    SQLAlchemy API for a complete behavioral description.

    .. versionadded:: 1.4

    """
    __slots__ = ()
    _generate_rows = False

    def __init__(self, real_result: Result[Any], index: _KeyIndexType):
        self._real_result = real_result
        if real_result._source_supports_scalars:
            self._metadata = real_result._metadata
            self._post_creational_filter = None
        else:
            self._metadata = real_result._metadata._reduce([index])
            self._post_creational_filter = operator.itemgetter(0)
        self._unique_filter_state = real_result._unique_filter_state

    def unique(self, strategy: Optional[_UniqueFilterType]=None) -> Self:
        """Apply unique filtering to the objects returned by this
        :class:`_asyncio.AsyncScalarResult`.

        See :meth:`_asyncio.AsyncResult.unique` for usage details.

        """
        self._unique_filter_state = (set(), strategy)
        return self

    async def partitions(self, size: Optional[int]=None) -> AsyncIterator[Sequence[_R]]:
        """Iterate through sub-lists of elements of the size given.

        Equivalent to :meth:`_asyncio.AsyncResult.partitions` except that
        scalar values, rather than :class:`_engine.Row` objects,
        are returned.

        """
        getter = self._manyrow_getter
        while True:
            partition = await greenlet_spawn(getter, self, size)
            if partition:
                yield partition
            else:
                break

    async def fetchall(self) -> Sequence[_R]:
        """A synonym for the :meth:`_asyncio.AsyncScalarResult.all` method."""
        return await greenlet_spawn(self._allrows)

    async def fetchmany(self, size: Optional[int]=None) -> Sequence[_R]:
        """Fetch many objects.

        Equivalent to :meth:`_asyncio.AsyncResult.fetchmany` except that
        scalar values, rather than :class:`_engine.Row` objects,
        are returned.

        """
        return await greenlet_spawn(self._manyrow_getter, self, size)

    async def all(self) -> Sequence[_R]:
        """Return all scalar values in a list.

        Equivalent to :meth:`_asyncio.AsyncResult.all` except that
        scalar values, rather than :class:`_engine.Row` objects,
        are returned.

        """
        return await greenlet_spawn(self._allrows)

    def __aiter__(self) -> AsyncScalarResult[_R]:
        return self

    async def __anext__(self) -> _R:
        row = await greenlet_spawn(self._onerow_getter, self)
        if row is _NO_ROW:
            raise StopAsyncIteration()
        else:
            return row

    async def first(self) -> Optional[_R]:
        """Fetch the first object or ``None`` if no object is present.

        Equivalent to :meth:`_asyncio.AsyncResult.first` except that
        scalar values, rather than :class:`_engine.Row` objects,
        are returned.

        """
        return await greenlet_spawn(self._only_one_row, False, False, False)

    async def one_or_none(self) -> Optional[_R]:
        """Return at most one object or raise an exception.

        Equivalent to :meth:`_asyncio.AsyncResult.one_or_none` except that
        scalar values, rather than :class:`_engine.Row` objects,
        are returned.

        """
        return await greenlet_spawn(self._only_one_row, True, False, False)

    async def one(self) -> _R:
        """Return exactly one object or raise an exception.

        Equivalent to :meth:`_asyncio.AsyncResult.one` except that
        scalar values, rather than :class:`_engine.Row` objects,
        are returned.

        """
        return await greenlet_spawn(self._only_one_row, True, True, False)