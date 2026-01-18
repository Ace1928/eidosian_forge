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
class _AssociationSet(_AssociationSingleItem[_T], MutableSet[_T]):
    """Generic, converting, set-to-set proxy."""
    col: MutableSet[_T]

    def __len__(self) -> int:
        return len(self.col)

    def __bool__(self) -> bool:
        if self.col:
            return True
        else:
            return False

    def __contains__(self, __o: object) -> bool:
        for member in self.col:
            if self._get(member) == __o:
                return True
        return False

    def __iter__(self) -> Iterator[_T]:
        """Iterate over proxied values.

        For the actual domain objects, iterate over .col instead or just use
        the underlying collection directly from its property on the parent.

        """
        for member in self.col:
            yield self._get(member)
        return

    def add(self, __element: _T) -> None:
        if __element not in self:
            self.col.add(self._create(__element))

    def discard(self, __element: _T) -> None:
        for member in self.col:
            if self._get(member) == __element:
                self.col.discard(member)
                break

    def remove(self, __element: _T) -> None:
        for member in self.col:
            if self._get(member) == __element:
                self.col.discard(member)
                return
        raise KeyError(__element)

    def pop(self) -> _T:
        if not self.col:
            raise KeyError('pop from an empty set')
        member = self.col.pop()
        return self._get(member)

    def update(self, *s: Iterable[_T]) -> None:
        for iterable in s:
            for value in iterable:
                self.add(value)

    def _bulk_replace(self, assoc_proxy: Any, values: Iterable[_T]) -> None:
        existing = set(self)
        constants = existing.intersection(values or ())
        additions = set(values or ()).difference(constants)
        removals = existing.difference(constants)
        appender = self.add
        remover = self.remove
        for member in values or ():
            if member in additions:
                appender(member)
            elif member in constants:
                appender(member)
        for member in removals:
            remover(member)

    def __ior__(self, other: AbstractSet[_S]) -> MutableSet[Union[_T, _S]]:
        if not collections._set_binops_check_strict(self, other):
            raise NotImplementedError()
        for value in other:
            self.add(value)
        return self

    def _set(self) -> Set[_T]:
        return set(iter(self))

    def union(self, *s: Iterable[_S]) -> MutableSet[Union[_T, _S]]:
        return set(self).union(*s)

    def __or__(self, __s: AbstractSet[_S]) -> MutableSet[Union[_T, _S]]:
        return self.union(__s)

    def difference(self, *s: Iterable[Any]) -> MutableSet[_T]:
        return set(self).difference(*s)

    def __sub__(self, s: AbstractSet[Any]) -> MutableSet[_T]:
        return self.difference(s)

    def difference_update(self, *s: Iterable[Any]) -> None:
        for other in s:
            for value in other:
                self.discard(value)

    def __isub__(self, s: AbstractSet[Any]) -> Self:
        if not collections._set_binops_check_strict(self, s):
            raise NotImplementedError()
        for value in s:
            self.discard(value)
        return self

    def intersection(self, *s: Iterable[Any]) -> MutableSet[_T]:
        return set(self).intersection(*s)

    def __and__(self, s: AbstractSet[Any]) -> MutableSet[_T]:
        return self.intersection(s)

    def intersection_update(self, *s: Iterable[Any]) -> None:
        for other in s:
            want, have = (self.intersection(other), set(self))
            remove, add = (have - want, want - have)
            for value in remove:
                self.remove(value)
            for value in add:
                self.add(value)

    def __iand__(self, s: AbstractSet[Any]) -> Self:
        if not collections._set_binops_check_strict(self, s):
            raise NotImplementedError()
        want = self.intersection(s)
        have: Set[_T] = set(self)
        remove, add = (have - want, want - have)
        for value in remove:
            self.remove(value)
        for value in add:
            self.add(value)
        return self

    def symmetric_difference(self, __s: Iterable[_T]) -> MutableSet[_T]:
        return set(self).symmetric_difference(__s)

    def __xor__(self, s: AbstractSet[_S]) -> MutableSet[Union[_T, _S]]:
        return self.symmetric_difference(s)

    def symmetric_difference_update(self, other: Iterable[Any]) -> None:
        want, have = (self.symmetric_difference(other), set(self))
        remove, add = (have - want, want - have)
        for value in remove:
            self.remove(value)
        for value in add:
            self.add(value)

    def __ixor__(self, other: AbstractSet[_S]) -> MutableSet[Union[_T, _S]]:
        if not collections._set_binops_check_strict(self, other):
            raise NotImplementedError()
        self.symmetric_difference_update(other)
        return self

    def issubset(self, __s: Iterable[Any]) -> bool:
        return set(self).issubset(__s)

    def issuperset(self, __s: Iterable[Any]) -> bool:
        return set(self).issuperset(__s)

    def clear(self) -> None:
        self.col.clear()

    def copy(self) -> AbstractSet[_T]:
        return set(self)

    def __eq__(self, other: object) -> bool:
        return set(self) == other

    def __ne__(self, other: object) -> bool:
        return set(self) != other

    def __lt__(self, other: AbstractSet[Any]) -> bool:
        return set(self) < other

    def __le__(self, other: AbstractSet[Any]) -> bool:
        return set(self) <= other

    def __gt__(self, other: AbstractSet[Any]) -> bool:
        return set(self) > other

    def __ge__(self, other: AbstractSet[Any]) -> bool:
        return set(self) >= other

    def __repr__(self) -> str:
        return repr(set(self))

    def __hash__(self) -> NoReturn:
        raise TypeError('%s objects are unhashable' % type(self).__name__)
    if not typing.TYPE_CHECKING:
        for func_name, func in list(locals().items()):
            if callable(func) and func.__name__ == func_name and (not func.__doc__) and hasattr(set, func_name):
                func.__doc__ = getattr(set, func_name).__doc__
        del func_name, func