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
class _AssociationDict(_AssociationCollection[_VT], MutableMapping[_KT, _VT]):
    """Generic, converting, dict-to-dict proxy."""
    setter: _DictSetterProtocol[_VT]
    creator: _KeyCreatorProtocol[_VT]
    col: MutableMapping[_KT, Optional[_VT]]

    def _create(self, key: _KT, value: Optional[_VT]) -> Any:
        return self.creator(key, value)

    def _get(self, object_: Any) -> _VT:
        return self.getter(object_)

    def _set(self, object_: Any, key: _KT, value: _VT) -> None:
        return self.setter(object_, key, value)

    def __getitem__(self, key: _KT) -> _VT:
        return self._get(self.col[key])

    def __setitem__(self, key: _KT, value: _VT) -> None:
        if key in self.col:
            self._set(self.col[key], key, value)
        else:
            self.col[key] = self._create(key, value)

    def __delitem__(self, key: _KT) -> None:
        del self.col[key]

    def __contains__(self, key: object) -> bool:
        return key in self.col

    def __iter__(self) -> Iterator[_KT]:
        return iter(self.col.keys())

    def clear(self) -> None:
        self.col.clear()

    def __eq__(self, other: object) -> bool:
        return dict(self) == other

    def __ne__(self, other: object) -> bool:
        return dict(self) != other

    def __repr__(self) -> str:
        return repr(dict(self))

    @overload
    def get(self, __key: _KT) -> Optional[_VT]:
        ...

    @overload
    def get(self, __key: _KT, default: Union[_VT, _T]) -> Union[_VT, _T]:
        ...

    def get(self, key: _KT, default: Optional[Union[_VT, _T]]=None) -> Union[_VT, _T, None]:
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key: _KT, default: Optional[_VT]=None) -> _VT:
        if key not in self.col:
            self.col[key] = self._create(key, default)
            return default
        else:
            return self[key]

    def keys(self) -> KeysView[_KT]:
        return self.col.keys()

    def items(self) -> ItemsView[_KT, _VT]:
        return ItemsView(self)

    def values(self) -> ValuesView[_VT]:
        return ValuesView(self)

    @overload
    def pop(self, __key: _KT) -> _VT:
        ...

    @overload
    def pop(self, __key: _KT, default: Union[_VT, _T]=...) -> Union[_VT, _T]:
        ...

    def pop(self, __key: _KT, *arg: Any, **kw: Any) -> Union[_VT, _T]:
        member = self.col.pop(__key, *arg, **kw)
        return self._get(member)

    def popitem(self) -> Tuple[_KT, _VT]:
        item = self.col.popitem()
        return (item[0], self._get(item[1]))

    @overload
    def update(self, __m: SupportsKeysAndGetItem[_KT, _VT], **kwargs: _VT) -> None:
        ...

    @overload
    def update(self, __m: Iterable[tuple[_KT, _VT]], **kwargs: _VT) -> None:
        ...

    @overload
    def update(self, **kwargs: _VT) -> None:
        ...

    def update(self, *a: Any, **kw: Any) -> None:
        up: Dict[_KT, _VT] = {}
        up.update(*a, **kw)
        for key, value in up.items():
            self[key] = value

    def _bulk_replace(self, assoc_proxy: AssociationProxyInstance[Any], values: Mapping[_KT, _VT]) -> None:
        existing = set(self)
        constants = existing.intersection(values or ())
        additions = set(values or ()).difference(constants)
        removals = existing.difference(constants)
        for key, member in values.items() or ():
            if key in additions:
                self[key] = member
            elif key in constants:
                self[key] = member
        for key in removals:
            del self[key]

    def copy(self) -> Dict[_KT, _VT]:
        return dict(self.items())

    def __hash__(self) -> NoReturn:
        raise TypeError('%s objects are unhashable' % type(self).__name__)
    if not typing.TYPE_CHECKING:
        for func_name, func in list(locals().items()):
            if callable(func) and func.__name__ == func_name and (not func.__doc__) and hasattr(dict, func_name):
                func.__doc__ = getattr(dict, func_name).__doc__
        del func_name, func