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
class IteratorResult(Result[_TP]):
    """A :class:`_engine.Result` that gets data from a Python iterator of
    :class:`_engine.Row` objects or similar row-like data.

    .. versionadded:: 1.4

    """
    _hard_closed = False
    _soft_closed = False

    def __init__(self, cursor_metadata: ResultMetaData, iterator: Iterator[_InterimSupportsScalarsRowType], raw: Optional[Result[Any]]=None, _source_supports_scalars: bool=False):
        self._metadata = cursor_metadata
        self.iterator = iterator
        self.raw = raw
        self._source_supports_scalars = _source_supports_scalars

    @property
    def closed(self) -> bool:
        """Return ``True`` if this :class:`_engine.IteratorResult` has
        been closed

        .. versionadded:: 1.4.43

        """
        return self._hard_closed

    def _soft_close(self, hard: bool=False, **kw: Any) -> None:
        if hard:
            self._hard_closed = True
        if self.raw is not None:
            self.raw._soft_close(hard=hard, **kw)
        self.iterator = iter([])
        self._reset_memoizations()
        self._soft_closed = True

    def _raise_hard_closed(self) -> NoReturn:
        raise exc.ResourceClosedError('This result object is closed.')

    def _raw_row_iterator(self) -> Iterator[_RowData]:
        return self.iterator

    def _fetchiter_impl(self) -> Iterator[_InterimSupportsScalarsRowType]:
        if self._hard_closed:
            self._raise_hard_closed()
        return self.iterator

    def _fetchone_impl(self, hard_close: bool=False) -> Optional[_InterimRowType[Row[Any]]]:
        if self._hard_closed:
            self._raise_hard_closed()
        row = next(self.iterator, _NO_ROW)
        if row is _NO_ROW:
            self._soft_close(hard=hard_close)
            return None
        else:
            return row

    def _fetchall_impl(self) -> List[_InterimRowType[Row[Any]]]:
        if self._hard_closed:
            self._raise_hard_closed()
        try:
            return list(self.iterator)
        finally:
            self._soft_close()

    def _fetchmany_impl(self, size: Optional[int]=None) -> List[_InterimRowType[Row[Any]]]:
        if self._hard_closed:
            self._raise_hard_closed()
        return list(itertools.islice(self.iterator, 0, size))