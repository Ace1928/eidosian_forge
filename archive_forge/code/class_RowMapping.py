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
class RowMapping(BaseRow, typing.Mapping['_KeyType', Any]):
    """A ``Mapping`` that maps column names and objects to :class:`.Row`
    values.

    The :class:`.RowMapping` is available from a :class:`.Row` via the
    :attr:`.Row._mapping` attribute, as well as from the iterable interface
    provided by the :class:`.MappingResult` object returned by the
    :meth:`_engine.Result.mappings` method.

    :class:`.RowMapping` supplies Python mapping (i.e. dictionary) access to
    the  contents of the row.   This includes support for testing of
    containment of specific keys (string column names or objects), as well
    as iteration of keys, values, and items::

        for row in result:
            if 'a' in row._mapping:
                print("Column 'a': %s" % row._mapping['a'])

            print("Column b: %s" % row._mapping[table.c.b])


    .. versionadded:: 1.4 The :class:`.RowMapping` object replaces the
       mapping-like access previously provided by a database result row,
       which now seeks to behave mostly like a named tuple.

    """
    __slots__ = ()
    if TYPE_CHECKING:

        def __getitem__(self, key: _KeyType) -> Any:
            ...
    else:
        __getitem__ = BaseRow._get_by_key_impl_mapping

    def _values_impl(self) -> List[Any]:
        return list(self._data)

    def __iter__(self) -> Iterator[str]:
        return (k for k in self._parent.keys if k is not None)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return self._parent._has_key(key)

    def __repr__(self) -> str:
        return repr(dict(self))

    def items(self) -> ROMappingItemsView:
        """Return a view of key/value tuples for the elements in the
        underlying :class:`.Row`.

        """
        return ROMappingItemsView(self, [(key, self[key]) for key in self.keys()])

    def keys(self) -> RMKeyView:
        """Return a view of 'keys' for string column names represented
        by the underlying :class:`.Row`.

        """
        return self._parent.keys

    def values(self) -> ROMappingKeysValuesView:
        """Return a view of values for the values represented in the
        underlying :class:`.Row`.

        """
        return ROMappingKeysValuesView(self, self._values_impl())