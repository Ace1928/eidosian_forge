from __future__ import annotations
from itertools import filterfalse
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from ..util.typing import Self
class IdentitySet:
    """A set that considers only object id() for uniqueness.

    This strategy has edge cases for builtin types- it's possible to have
    two 'foo' strings in one of these sets, for example.  Use sparingly.

    """
    _members: Dict[int, Any]

    def __init__(self, iterable: Optional[Iterable[Any]]=None):
        self._members = dict()
        if iterable:
            self.update(iterable)

    def add(self, value: Any) -> None:
        self._members[id(value)] = value

    def __contains__(self, value: Any) -> bool:
        return id(value) in self._members

    def remove(self, value: Any) -> None:
        del self._members[id(value)]

    def discard(self, value: Any) -> None:
        try:
            self.remove(value)
        except KeyError:
            pass

    def pop(self) -> Any:
        try:
            pair = self._members.popitem()
            return pair[1]
        except KeyError:
            raise KeyError('pop from an empty set')

    def clear(self) -> None:
        self._members.clear()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, IdentitySet):
            return self._members == other._members
        else:
            return False

    def __ne__(self, other: Any) -> bool:
        if isinstance(other, IdentitySet):
            return self._members != other._members
        else:
            return True

    def issubset(self, iterable: Iterable[Any]) -> bool:
        if isinstance(iterable, self.__class__):
            other = iterable
        else:
            other = self.__class__(iterable)
        if len(self) > len(other):
            return False
        for m in filterfalse(other._members.__contains__, iter(self._members.keys())):
            return False
        return True

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return self.issubset(other)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return len(self) < len(other) and self.issubset(other)

    def issuperset(self, iterable: Iterable[Any]) -> bool:
        if isinstance(iterable, self.__class__):
            other = iterable
        else:
            other = self.__class__(iterable)
        if len(self) < len(other):
            return False
        for m in filterfalse(self._members.__contains__, iter(other._members.keys())):
            return False
        return True

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return self.issuperset(other)

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return len(self) > len(other) and self.issuperset(other)

    def union(self, iterable: Iterable[Any]) -> IdentitySet:
        result = self.__class__()
        members = self._members
        result._members.update(members)
        result._members.update(((id(obj), obj) for obj in iterable))
        return result

    def __or__(self, other: Any) -> IdentitySet:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return self.union(other)

    def update(self, iterable: Iterable[Any]) -> None:
        self._members.update(((id(obj), obj) for obj in iterable))

    def __ior__(self, other: Any) -> IdentitySet:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        self.update(other)
        return self

    def difference(self, iterable: Iterable[Any]) -> IdentitySet:
        result = self.__new__(self.__class__)
        other: Collection[Any]
        if isinstance(iterable, self.__class__):
            other = iterable._members
        else:
            other = {id(obj) for obj in iterable}
        result._members = {k: v for k, v in self._members.items() if k not in other}
        return result

    def __sub__(self, other: IdentitySet) -> IdentitySet:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return self.difference(other)

    def difference_update(self, iterable: Iterable[Any]) -> None:
        self._members = self.difference(iterable)._members

    def __isub__(self, other: IdentitySet) -> IdentitySet:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        self.difference_update(other)
        return self

    def intersection(self, iterable: Iterable[Any]) -> IdentitySet:
        result = self.__new__(self.__class__)
        other: Collection[Any]
        if isinstance(iterable, self.__class__):
            other = iterable._members
        else:
            other = {id(obj) for obj in iterable}
        result._members = {k: v for k, v in self._members.items() if k in other}
        return result

    def __and__(self, other: IdentitySet) -> IdentitySet:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return self.intersection(other)

    def intersection_update(self, iterable: Iterable[Any]) -> None:
        self._members = self.intersection(iterable)._members

    def __iand__(self, other: IdentitySet) -> IdentitySet:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        self.intersection_update(other)
        return self

    def symmetric_difference(self, iterable: Iterable[Any]) -> IdentitySet:
        result = self.__new__(self.__class__)
        if isinstance(iterable, self.__class__):
            other = iterable._members
        else:
            other = {id(obj): obj for obj in iterable}
        result._members = {k: v for k, v in self._members.items() if k not in other}
        result._members.update(((k, v) for k, v in other.items() if k not in self._members))
        return result

    def __xor__(self, other: IdentitySet) -> IdentitySet:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        return self.symmetric_difference(other)

    def symmetric_difference_update(self, iterable: Iterable[Any]) -> None:
        self._members = self.symmetric_difference(iterable)._members

    def __ixor__(self, other: IdentitySet) -> IdentitySet:
        if not isinstance(other, IdentitySet):
            return NotImplemented
        self.symmetric_difference(other)
        return self

    def copy(self) -> IdentitySet:
        result = self.__new__(self.__class__)
        result._members = self._members.copy()
        return result
    __copy__ = copy

    def __len__(self) -> int:
        return len(self._members)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._members.values())

    def __hash__(self) -> NoReturn:
        raise TypeError('set objects are unhashable')

    def __repr__(self) -> str:
        return '%s(%r)' % (type(self).__name__, list(self._members.values()))