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