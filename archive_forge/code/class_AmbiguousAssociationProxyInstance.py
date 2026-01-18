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
class AmbiguousAssociationProxyInstance(AssociationProxyInstance[_T]):
    """an :class:`.AssociationProxyInstance` where we cannot determine
    the type of target object.
    """
    _is_canonical = False

    def _ambiguous(self) -> NoReturn:
        raise AttributeError("Association proxy %s.%s refers to an attribute '%s' that is not directly mapped on class %s; therefore this operation cannot proceed since we don't know what type of object is referred towards" % (self.owning_class.__name__, self.target_collection, self.value_attr, self.target_class))

    def get(self, obj: Any) -> Any:
        if obj is None:
            return self
        else:
            return super().get(obj)

    def __eq__(self, obj: object) -> NoReturn:
        self._ambiguous()

    def __ne__(self, obj: object) -> NoReturn:
        self._ambiguous()

    def any(self, criterion: Optional[_ColumnExpressionArgument[bool]]=None, **kwargs: Any) -> NoReturn:
        self._ambiguous()

    def has(self, criterion: Optional[_ColumnExpressionArgument[bool]]=None, **kwargs: Any) -> NoReturn:
        self._ambiguous()

    @util.memoized_property
    def _lookup_cache(self) -> Dict[Type[Any], AssociationProxyInstance[_T]]:
        return {}

    def _non_canonical_get_for_object(self, parent_instance: Any) -> AssociationProxyInstance[_T]:
        if parent_instance is not None:
            actual_obj = getattr(parent_instance, self.target_collection)
            if actual_obj is not None:
                try:
                    insp = inspect(actual_obj)
                except exc.NoInspectionAvailable:
                    pass
                else:
                    mapper = insp.mapper
                    instance_class = mapper.class_
                    if instance_class not in self._lookup_cache:
                        self._populate_cache(instance_class, mapper)
                    try:
                        return self._lookup_cache[instance_class]
                    except KeyError:
                        pass
        return self

    def _populate_cache(self, instance_class: Any, mapper: Mapper[Any]) -> None:
        prop = orm.class_mapper(self.owning_class).get_property(self.target_collection)
        if mapper.isa(prop.mapper):
            target_class = instance_class
            try:
                target_assoc = self._cls_unwrap_target_assoc_proxy(target_class, self.value_attr)
            except AttributeError:
                pass
            else:
                self._lookup_cache[instance_class] = self._construct_for_assoc(cast('AssociationProxyInstance[_T]', target_assoc), self.parent, self.owning_class, target_class, self.value_attr)