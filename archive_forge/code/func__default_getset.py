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
def _default_getset(self, collection_class: Any) -> Tuple[_GetterProtocol[Any], _SetterProtocol]:
    attr = self.value_attr
    _getter = operator.attrgetter(attr)

    def getter(instance: Any) -> Optional[_T]:
        return _getter(instance) if instance is not None else None
    if collection_class is dict:

        def dict_setter(instance: Any, k: Any, value: _T) -> None:
            setattr(instance, attr, value)
        return (getter, dict_setter)
    else:

        def plain_setter(o: Any, v: _T) -> None:
            setattr(o, attr, v)
        return (getter, plain_setter)