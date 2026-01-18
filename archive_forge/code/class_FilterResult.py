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
class FilterResult(ResultInternal[_R]):
    """A wrapper for a :class:`_engine.Result` that returns objects other than
    :class:`_engine.Row` objects, such as dictionaries or scalar objects.

    :class:`_engine.FilterResult` is the common base for additional result
    APIs including :class:`_engine.MappingResult`,
    :class:`_engine.ScalarResult` and :class:`_engine.AsyncResult`.

    """
    __slots__ = ('_real_result', '_post_creational_filter', '_metadata', '_unique_filter_state', '__dict__')
    _post_creational_filter: Optional[Callable[[Any], Any]]
    _real_result: Result[Any]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, type_: Any, value: Any, traceback: Any) -> None:
        self._real_result.__exit__(type_, value, traceback)

    @_generative
    def yield_per(self, num: int) -> Self:
        """Configure the row-fetching strategy to fetch ``num`` rows at a time.

        The :meth:`_engine.FilterResult.yield_per` method is a pass through
        to the :meth:`_engine.Result.yield_per` method.  See that method's
        documentation for usage notes.

        .. versionadded:: 1.4.40 - added :meth:`_engine.FilterResult.yield_per`
           so that the method is available on all result set implementations

        .. seealso::

            :ref:`engine_stream_results` - describes Core behavior for
            :meth:`_engine.Result.yield_per`

            :ref:`orm_queryguide_yield_per` - in the :ref:`queryguide_toplevel`

        """
        self._real_result = self._real_result.yield_per(num)
        return self

    def _soft_close(self, hard: bool=False) -> None:
        self._real_result._soft_close(hard=hard)

    @property
    def _soft_closed(self) -> bool:
        return self._real_result._soft_closed

    @property
    def closed(self) -> bool:
        """Return ``True`` if the underlying :class:`_engine.Result` reports
        closed

        .. versionadded:: 1.4.43

        """
        return self._real_result.closed

    def close(self) -> None:
        """Close this :class:`_engine.FilterResult`.

        .. versionadded:: 1.4.43

        """
        self._real_result.close()

    @property
    def _attributes(self) -> Dict[Any, Any]:
        return self._real_result._attributes

    def _fetchiter_impl(self) -> Iterator[_InterimRowType[Row[Any]]]:
        return self._real_result._fetchiter_impl()

    def _fetchone_impl(self, hard_close: bool=False) -> Optional[_InterimRowType[Row[Any]]]:
        return self._real_result._fetchone_impl(hard_close=hard_close)

    def _fetchall_impl(self) -> List[_InterimRowType[Row[Any]]]:
        return self._real_result._fetchall_impl()

    def _fetchmany_impl(self, size: Optional[int]=None) -> List[_InterimRowType[Row[Any]]]:
        return self._real_result._fetchmany_impl(size=size)