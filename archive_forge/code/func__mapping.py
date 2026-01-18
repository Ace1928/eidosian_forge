from __future__ import annotations
from abc import ABC
import collections.abc as collections_abc
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from ..sql import util as sql_util
from ..util import deprecated
from ..util._has_cy import HAS_CYEXTENSION
@property
def _mapping(self) -> RowMapping:
    """Return a :class:`.RowMapping` for this :class:`.Row`.

        This object provides a consistent Python mapping (i.e. dictionary)
        interface for the data contained within the row.   The :class:`.Row`
        by itself behaves like a named tuple.

        .. seealso::

            :attr:`.Row._fields`

        .. versionadded:: 1.4

        """
    return RowMapping(self._parent, None, self._key_to_index, self._data)