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
class _AssociationCollection(Generic[_IT]):
    getter: _GetterProtocol[_IT]
    "A function.  Given an associated object, return the 'value'."
    creator: _CreatorProtocol
    '\n    A function that creates new target entities.  Given one parameter:\n    value.  This assertion is assumed::\n\n    obj = creator(somevalue)\n    assert getter(obj) == somevalue\n    '
    parent: AssociationProxyInstance[_IT]
    setter: _SetterProtocol
    'A function.  Given an associated object and a value, store that\n        value on the object.\n    '
    lazy_collection: _LazyCollectionProtocol[_IT]
    'A callable returning a list-based collection of entities (usually an\n          object attribute managed by a SQLAlchemy relationship())'

    def __init__(self, lazy_collection: _LazyCollectionProtocol[_IT], creator: _CreatorProtocol, getter: _GetterProtocol[_IT], setter: _SetterProtocol, parent: AssociationProxyInstance[_IT]):
        """Constructs an _AssociationCollection.

        This will always be a subclass of either _AssociationList,
        _AssociationSet, or _AssociationDict.

        """
        self.lazy_collection = lazy_collection
        self.creator = creator
        self.getter = getter
        self.setter = setter
        self.parent = parent
    if typing.TYPE_CHECKING:
        col: Collection[_IT]
    else:
        col = property(lambda self: self.lazy_collection())

    def __len__(self) -> int:
        return len(self.col)

    def __bool__(self) -> bool:
        return bool(self.col)

    def __getstate__(self) -> Any:
        return {'parent': self.parent, 'lazy_collection': self.lazy_collection}

    def __setstate__(self, state: Any) -> None:
        self.parent = state['parent']
        self.lazy_collection = state['lazy_collection']
        self.parent._inflate(self)

    def clear(self) -> None:
        raise NotImplementedError()