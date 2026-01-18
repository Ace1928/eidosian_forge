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
class ObjectAssociationProxyInstance(AssociationProxyInstance[_T]):
    """an :class:`.AssociationProxyInstance` that has an object as a target."""
    _target_is_object: bool = True
    _is_canonical = True

    def contains(self, other: Any, **kw: Any) -> ColumnElement[bool]:
        """Produce a proxied 'contains' expression using EXISTS.

        This expression will be a composed product
        using the :meth:`.Relationship.Comparator.any`,
        :meth:`.Relationship.Comparator.has`,
        and/or :meth:`.Relationship.Comparator.contains`
        operators of the underlying proxied attributes.
        """
        target_assoc = self._unwrap_target_assoc_proxy
        if target_assoc is not None:
            return self._comparator._criterion_exists(target_assoc.contains(other) if not target_assoc.scalar else target_assoc == other)
        elif self._target_is_object and self.scalar and (not self._value_is_scalar):
            return self._comparator.has(getattr(self.target_class, self.value_attr).contains(other))
        elif self._target_is_object and self.scalar and self._value_is_scalar:
            raise exc.InvalidRequestError("contains() doesn't apply to a scalar object endpoint; use ==")
        else:
            return self._comparator._criterion_exists(**{self.value_attr: other})

    def __eq__(self, obj: Any) -> ColumnElement[bool]:
        if obj is None:
            return or_(self._comparator.has(**{self.value_attr: obj}), self._comparator == None)
        else:
            return self._comparator.has(**{self.value_attr: obj})

    def __ne__(self, obj: Any) -> ColumnElement[bool]:
        return self._comparator.has(getattr(self.target_class, self.value_attr) != obj)