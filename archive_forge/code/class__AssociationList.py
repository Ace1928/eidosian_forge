from __future__ import annotations
import operator
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Generic
from typing import ItemsView
from typing import Iterable
from typing import Iterator
from typing import KeysView
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import MutableSequence
from typing import MutableSet
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from typing import ValuesView
from .. import ColumnElement
from .. import exc
from .. import inspect
from .. import orm
from .. import util
from ..orm import collections
from ..orm import InspectionAttrExtensionType
from ..orm import interfaces
from ..orm import ORMDescriptor
from ..orm.base import SQLORMOperations
from ..orm.interfaces import _AttributeOptions
from ..orm.interfaces import _DCAttributeOptions
from ..orm.interfaces import _DEFAULT_ATTRIBUTE_OPTIONS
from ..sql import operators
from ..sql import or_
from ..sql.base import _NoArg
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import SupportsIndex
from ..util.typing import SupportsKeysAndGetItem
class _AssociationList(_AssociationSingleItem[_T], MutableSequence[_T]):
    """Generic, converting, list-to-list proxy."""
    col: MutableSequence[_T]

    def _set(self, object_: Any, value: _T) -> None:
        self.setter(object_, value)

    @overload
    def __getitem__(self, index: int) -> _T:
        ...

    @overload
    def __getitem__(self, index: slice) -> MutableSequence[_T]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[_T, MutableSequence[_T]]:
        if not isinstance(index, slice):
            return self._get(self.col[index])
        else:
            return [self._get(member) for member in self.col[index]]

    @overload
    def __setitem__(self, index: int, value: _T) -> None:
        ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[_T]) -> None:
        ...

    def __setitem__(self, index: Union[int, slice], value: Union[_T, Iterable[_T]]) -> None:
        if not isinstance(index, slice):
            self._set(self.col[index], cast('_T', value))
        else:
            if index.stop is None:
                stop = len(self)
            elif index.stop < 0:
                stop = len(self) + index.stop
            else:
                stop = index.stop
            step = index.step or 1
            start = index.start or 0
            rng = list(range(index.start or 0, stop, step))
            sized_value = list(value)
            if step == 1:
                for i in rng:
                    del self[start]
                i = start
                for item in sized_value:
                    self.insert(i, item)
                    i += 1
            else:
                if len(sized_value) != len(rng):
                    raise ValueError('attempt to assign sequence of size %s to extended slice of size %s' % (len(sized_value), len(rng)))
                for i, item in zip(rng, value):
                    self._set(self.col[i], item)

    @overload
    def __delitem__(self, index: int) -> None:
        ...

    @overload
    def __delitem__(self, index: slice) -> None:
        ...

    def __delitem__(self, index: Union[slice, int]) -> None:
        del self.col[index]

    def __contains__(self, value: object) -> bool:
        for member in self.col:
            if self._get(member) == value:
                return True
        return False

    def __iter__(self) -> Iterator[_T]:
        """Iterate over proxied values.

        For the actual domain objects, iterate over .col instead or
        just use the underlying collection directly from its property
        on the parent.
        """
        for member in self.col:
            yield self._get(member)
        return

    def append(self, value: _T) -> None:
        col = self.col
        item = self._create(value)
        col.append(item)

    def count(self, value: Any) -> int:
        count = 0
        for v in self:
            if v == value:
                count += 1
        return count

    def extend(self, values: Iterable[_T]) -> None:
        for v in values:
            self.append(v)

    def insert(self, index: int, value: _T) -> None:
        self.col[index:index] = [self._create(value)]

    def pop(self, index: int=-1) -> _T:
        return self.getter(self.col.pop(index))

    def remove(self, value: _T) -> None:
        for i, val in enumerate(self):
            if val == value:
                del self.col[i]
                return
        raise ValueError('value not in list')

    def reverse(self) -> NoReturn:
        """Not supported, use reversed(mylist)"""
        raise NotImplementedError()

    def sort(self) -> NoReturn:
        """Not supported, use sorted(mylist)"""
        raise NotImplementedError()

    def clear(self) -> None:
        del self.col[0:len(self.col)]

    def __eq__(self, other: object) -> bool:
        return list(self) == other

    def __ne__(self, other: object) -> bool:
        return list(self) != other

    def __lt__(self, other: List[_T]) -> bool:
        return list(self) < other

    def __le__(self, other: List[_T]) -> bool:
        return list(self) <= other

    def __gt__(self, other: List[_T]) -> bool:
        return list(self) > other

    def __ge__(self, other: List[_T]) -> bool:
        return list(self) >= other

    def __add__(self, other: List[_T]) -> List[_T]:
        try:
            other = list(other)
        except TypeError:
            return NotImplemented
        return list(self) + other

    def __radd__(self, other: List[_T]) -> List[_T]:
        try:
            other = list(other)
        except TypeError:
            return NotImplemented
        return other + list(self)

    def __mul__(self, n: SupportsIndex) -> List[_T]:
        if not isinstance(n, int):
            return NotImplemented
        return list(self) * n

    def __rmul__(self, n: SupportsIndex) -> List[_T]:
        if not isinstance(n, int):
            return NotImplemented
        return n * list(self)

    def __iadd__(self, iterable: Iterable[_T]) -> Self:
        self.extend(iterable)
        return self

    def __imul__(self, n: SupportsIndex) -> Self:
        if not isinstance(n, int):
            raise NotImplementedError()
        if n == 0:
            self.clear()
        elif n > 1:
            self.extend(list(self) * (n - 1))
        return self
    if typing.TYPE_CHECKING:

        def index(self, value: Any, start: int=..., stop: int=...) -> int:
            ...
    else:

        def index(self, value: Any, *arg) -> int:
            ls = list(self)
            return ls.index(value, *arg)

    def copy(self) -> List[_T]:
        return list(self)

    def __repr__(self) -> str:
        return repr(list(self))

    def __hash__(self) -> NoReturn:
        raise TypeError('%s objects are unhashable' % type(self).__name__)
    if not typing.TYPE_CHECKING:
        for func_name, func in list(locals().items()):
            if callable(func) and func.__name__ == func_name and (not func.__doc__) and hasattr(list, func_name):
                func.__doc__ = getattr(list, func_name).__doc__
        del func_name, func