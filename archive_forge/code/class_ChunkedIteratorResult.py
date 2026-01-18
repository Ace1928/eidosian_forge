from __future__ import annotations
from enum import Enum
import functools
import itertools
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .row import Row
from .row import RowMapping
from .. import exc
from .. import util
from ..sql.base import _generative
from ..sql.base import HasMemoized
from ..sql.base import InPlaceGenerative
from ..util import HasMemoized_ro_memoized_attribute
from ..util import NONE_SET
from ..util._has_cy import HAS_CYEXTENSION
from ..util.typing import Literal
from ..util.typing import Self
class ChunkedIteratorResult(IteratorResult[_TP]):
    """An :class:`_engine.IteratorResult` that works from an
    iterator-producing callable.

    The given ``chunks`` argument is a function that is given a number of rows
    to return in each chunk, or ``None`` for all rows.  The function should
    then return an un-consumed iterator of lists, each list of the requested
    size.

    The function can be called at any time again, in which case it should
    continue from the same result set but adjust the chunk size as given.

    .. versionadded:: 1.4

    """

    def __init__(self, cursor_metadata: ResultMetaData, chunks: Callable[[Optional[int]], Iterator[Sequence[_InterimRowType[_R]]]], source_supports_scalars: bool=False, raw: Optional[Result[Any]]=None, dynamic_yield_per: bool=False):
        self._metadata = cursor_metadata
        self.chunks = chunks
        self._source_supports_scalars = source_supports_scalars
        self.raw = raw
        self.iterator = itertools.chain.from_iterable(self.chunks(None))
        self.dynamic_yield_per = dynamic_yield_per

    @_generative
    def yield_per(self, num: int) -> Self:
        self._yield_per = num
        self.iterator = itertools.chain.from_iterable(self.chunks(num))
        return self

    def _soft_close(self, hard: bool=False, **kw: Any) -> None:
        super()._soft_close(hard=hard, **kw)
        self.chunks = lambda size: []

    def _fetchmany_impl(self, size: Optional[int]=None) -> List[_InterimRowType[Row[Any]]]:
        if self.dynamic_yield_per:
            self.iterator = itertools.chain.from_iterable(self.chunks(size))
        return super()._fetchmany_impl(size=size)