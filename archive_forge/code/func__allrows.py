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