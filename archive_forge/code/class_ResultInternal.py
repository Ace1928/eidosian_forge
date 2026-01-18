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
class ResultInternal(InPlaceGenerative, Generic[_R]):
    __slots__ = ()
    _real_result: Optional[Result[Any]] = None
    _generate_rows: bool = True
    _row_logging_fn: Optional[Callable[[Any], Any]]
    _unique_filter_state: Optional[_UniqueFilterStateType] = None
    _post_creational_filter: Optional[Callable[[Any], Any]] = None
    _is_cursor = False
    _metadata: ResultMetaData
    _source_supports_scalars: bool

    def _fetchiter_impl(self) -> Iterator[_InterimRowType[Row[Any]]]:
        raise NotImplementedError()

    def _fetchone_impl(self, hard_close: bool=False) -> Optional[_InterimRowType[Row[Any]]]:
        raise NotImplementedError()

    def _fetchmany_impl(self, size: Optional[int]=None) -> List[_InterimRowType[Row[Any]]]:
        raise NotImplementedError()

    def _fetchall_impl(self) -> List[_InterimRowType[Row[Any]]]:
        raise NotImplementedError()

    def _soft_close(self, hard: bool=False) -> None:
        raise NotImplementedError()

    @HasMemoized_ro_memoized_attribute
    def _row_getter(self) -> Optional[Callable[..., _R]]:
        real_result: Result[Any] = self._real_result if self._real_result else cast('Result[Any]', self)
        if real_result._source_supports_scalars:
            if not self._generate_rows:
                return None
            else:
                _proc = Row

                def process_row(metadata: ResultMetaData, processors: Optional[_ProcessorsType], key_to_index: Mapping[_KeyType, int], scalar_obj: Any) -> Row[Any]:
                    return _proc(metadata, processors, key_to_index, (scalar_obj,))
        else:
            process_row = Row
        metadata = self._metadata
        key_to_index = metadata._key_to_index
        processors = metadata._effective_processors
        tf = metadata._tuplefilter
        if tf and (not real_result._source_supports_scalars):
            if processors:
                processors = tf(processors)
            _make_row_orig: Callable[..., _R] = functools.partial(process_row, metadata, processors, key_to_index)
            fixed_tf = tf

            def make_row(row: _InterimRowType[Row[Any]]) -> _R:
                return _make_row_orig(fixed_tf(row))
        else:
            make_row = functools.partial(process_row, metadata, processors, key_to_index)
        if real_result._row_logging_fn:
            _log_row = real_result._row_logging_fn
            _make_row = make_row

            def make_row(row: _InterimRowType[Row[Any]]) -> _R:
                return _log_row(_make_row(row))
        return make_row

    @HasMemoized_ro_memoized_attribute
    def _iterator_getter(self) -> Callable[..., Iterator[_R]]:
        make_row = self._row_getter
        post_creational_filter = self._post_creational_filter
        if self._unique_filter_state:
            uniques, strategy = self._unique_strategy

            def iterrows(self: Result[Any]) -> Iterator[_R]:
                for raw_row in self._fetchiter_impl():
                    obj: _InterimRowType[Any] = make_row(raw_row) if make_row else raw_row
                    hashed = strategy(obj) if strategy else obj
                    if hashed in uniques:
                        continue
                    uniques.add(hashed)
                    if post_creational_filter:
                        obj = post_creational_filter(obj)
                    yield obj
        else:

            def iterrows(self: Result[Any]) -> Iterator[_R]:
                for raw_row in self._fetchiter_impl():
                    row: _InterimRowType[Any] = make_row(raw_row) if make_row else raw_row
                    if post_creational_filter:
                        row = post_creational_filter(row)
                    yield row
        return iterrows

    def _raw_all_rows(self) -> List[_R]:
        make_row = self._row_getter
        assert make_row is not None
        rows = self._fetchall_impl()
        return [make_row(row) for row in rows]

    def _allrows(self) -> List[_R]:
        post_creational_filter = self._post_creational_filter
        make_row = self._row_getter
        rows = self._fetchall_impl()
        made_rows: List[_InterimRowType[_R]]
        if make_row:
            made_rows = [make_row(row) for row in rows]
        else:
            made_rows = rows
        interim_rows: List[_R]
        if self._unique_filter_state:
            uniques, strategy = self._unique_strategy
            interim_rows = [made_row for made_row, sig_row in [(made_row, strategy(made_row) if strategy else made_row) for made_row in made_rows] if sig_row not in uniques and (not uniques.add(sig_row))]
        else:
            interim_rows = made_rows
        if post_creational_filter:
            interim_rows = [post_creational_filter(row) for row in interim_rows]
        return interim_rows

    @HasMemoized_ro_memoized_attribute
    def _onerow_getter(self) -> Callable[..., Union[Literal[_NoRow._NO_ROW], _R]]:
        make_row = self._row_getter
        post_creational_filter = self._post_creational_filter
        if self._unique_filter_state:
            uniques, strategy = self._unique_strategy

            def onerow(self: Result[Any]) -> Union[_NoRow, _R]:
                _onerow = self._fetchone_impl
                while True:
                    row = _onerow()
                    if row is None:
                        return _NO_ROW
                    else:
                        obj: _InterimRowType[Any] = make_row(row) if make_row else row
                        hashed = strategy(obj) if strategy else obj
                        if hashed in uniques:
                            continue
                        else:
                            uniques.add(hashed)
                        if post_creational_filter:
                            obj = post_creational_filter(obj)
                        return obj
        else:

            def onerow(self: Result[Any]) -> Union[_NoRow, _R]:
                row = self._fetchone_impl()
                if row is None:
                    return _NO_ROW
                else:
                    interim_row: _InterimRowType[Any] = make_row(row) if make_row else row
                    if post_creational_filter:
                        interim_row = post_creational_filter(interim_row)
                    return interim_row
        return onerow

    @HasMemoized_ro_memoized_attribute
    def _manyrow_getter(self) -> Callable[..., List[_R]]:
        make_row = self._row_getter
        post_creational_filter = self._post_creational_filter
        if self._unique_filter_state:
            uniques, strategy = self._unique_strategy

            def filterrows(make_row: Optional[Callable[..., _R]], rows: List[Any], strategy: Optional[Callable[[List[Any]], Any]], uniques: Set[Any]) -> List[_R]:
                if make_row:
                    rows = [make_row(row) for row in rows]
                if strategy:
                    made_rows = ((made_row, strategy(made_row)) for made_row in rows)
                else:
                    made_rows = ((made_row, made_row) for made_row in rows)
                return [made_row for made_row, sig_row in made_rows if sig_row not in uniques and (not uniques.add(sig_row))]

            def manyrows(self: ResultInternal[_R], num: Optional[int]) -> List[_R]:
                collect: List[_R] = []
                _manyrows = self._fetchmany_impl
                if num is None:
                    real_result = self._real_result if self._real_result else cast('Result[Any]', self)
                    if real_result._yield_per:
                        num_required = num = real_result._yield_per
                    else:
                        rows = _manyrows(num)
                        num = len(rows)
                        assert make_row is not None
                        collect.extend(filterrows(make_row, rows, strategy, uniques))
                        num_required = num - len(collect)
                else:
                    num_required = num
                assert num is not None
                while num_required:
                    rows = _manyrows(num_required)
                    if not rows:
                        break
                    collect.extend(filterrows(make_row, rows, strategy, uniques))
                    num_required = num - len(collect)
                if post_creational_filter:
                    collect = [post_creational_filter(row) for row in collect]
                return collect
        else:

            def manyrows(self: ResultInternal[_R], num: Optional[int]) -> List[_R]:
                if num is None:
                    real_result = self._real_result if self._real_result else cast('Result[Any]', self)
                    num = real_result._yield_per
                rows: List[_InterimRowType[Any]] = self._fetchmany_impl(num)
                if make_row:
                    rows = [make_row(row) for row in rows]
                if post_creational_filter:
                    rows = [post_creational_filter(row) for row in rows]
                return rows
        return manyrows

    @overload
    def _only_one_row(self, raise_for_second_row: bool, raise_for_none: Literal[True], scalar: bool) -> _R:
        ...

    @overload
    def _only_one_row(self, raise_for_second_row: bool, raise_for_none: bool, scalar: bool) -> Optional[_R]:
        ...

    def _only_one_row(self, raise_for_second_row: bool, raise_for_none: bool, scalar: bool) -> Optional[_R]:
        onerow = self._fetchone_impl
        row: Optional[_InterimRowType[Any]] = onerow(hard_close=True)
        if row is None:
            if raise_for_none:
                raise exc.NoResultFound('No row was found when one was required')
            else:
                return None
        if scalar and self._source_supports_scalars:
            self._generate_rows = False
            make_row = None
        else:
            make_row = self._row_getter
        try:
            row = make_row(row) if make_row else row
        except:
            self._soft_close(hard=True)
            raise
        if raise_for_second_row:
            if self._unique_filter_state:
                uniques, strategy = self._unique_strategy
                existing_row_hash = strategy(row) if strategy else row
                while True:
                    next_row: Any = onerow(hard_close=True)
                    if next_row is None:
                        next_row = _NO_ROW
                        break
                    try:
                        next_row = make_row(next_row) if make_row else next_row
                        if strategy:
                            assert next_row is not _NO_ROW
                            if existing_row_hash == strategy(next_row):
                                continue
                        elif row == next_row:
                            continue
                        break
                    except:
                        self._soft_close(hard=True)
                        raise
            else:
                next_row = onerow(hard_close=True)
                if next_row is None:
                    next_row = _NO_ROW
            if next_row is not _NO_ROW:
                self._soft_close(hard=True)
                raise exc.MultipleResultsFound('Multiple rows were found when exactly one was required' if raise_for_none else 'Multiple rows were found when one or none was required')
        else:
            next_row = _NO_ROW
            self._soft_close(hard=True)
        if not scalar:
            post_creational_filter = self._post_creational_filter
            if post_creational_filter:
                row = post_creational_filter(row)
        if scalar and make_row:
            return row[0]
        else:
            return row

    def _iter_impl(self) -> Iterator[_R]:
        return self._iterator_getter(self)

    def _next_impl(self) -> _R:
        row = self._onerow_getter(self)
        if row is _NO_ROW:
            raise StopIteration()
        else:
            return row

    @_generative
    def _column_slices(self, indexes: Sequence[_KeyIndexType]) -> Self:
        real_result = self._real_result if self._real_result else cast('Result[Any]', self)
        if not real_result._source_supports_scalars or len(indexes) != 1:
            self._metadata = self._metadata._reduce(indexes)
        assert self._generate_rows
        return self

    @HasMemoized.memoized_attribute
    def _unique_strategy(self) -> _UniqueFilterStateType:
        assert self._unique_filter_state is not None
        uniques, strategy = self._unique_filter_state
        real_result = self._real_result if self._real_result is not None else cast('Result[Any]', self)
        if not strategy and self._metadata._unique_filters:
            if real_result._source_supports_scalars and (not self._generate_rows):
                strategy = self._metadata._unique_filters[0]
            else:
                filters = self._metadata._unique_filters
                if self._metadata._tuplefilter:
                    filters = self._metadata._tuplefilter(filters)
                strategy = operator.methodcaller('_filter_on_values', filters)
        return (uniques, strategy)