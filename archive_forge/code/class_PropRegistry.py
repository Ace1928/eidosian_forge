from __future__ import annotations
from functools import reduce
from itertools import chain
import logging
import operator
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import base as orm_base
from ._typing import insp_is_mapper_property
from .. import exc
from .. import util
from ..sql import visitors
from ..sql.cache_key import HasCacheKey
class PropRegistry(PathRegistry):
    __slots__ = ('prop', 'parent', 'path', 'natural_path', 'has_entity', 'entity', 'mapper', '_wildcard_path_loader_key', '_default_path_loader_key', '_loader_key', 'is_unnatural')
    inherit_cache = True
    is_property = True
    prop: MapperProperty[Any]
    mapper: Optional[Mapper[Any]]
    entity: Optional[_InternalEntityType[Any]]

    def __init__(self, parent: AbstractEntityRegistry, prop: MapperProperty[Any]):
        insp = cast('_InternalEntityType[Any]', parent[-1])
        natural_parent: AbstractEntityRegistry = parent
        self.is_unnatural = parent.parent.is_unnatural or bool(parent.mapper.inherits)
        if not insp.is_aliased_class or insp._use_mapper_path:
            parent = natural_parent = parent.parent[prop.parent]
        elif insp.is_aliased_class and insp.with_polymorphic_mappers and (prop.parent in insp.with_polymorphic_mappers):
            subclass_entity: _InternalEntityType[Any] = parent[-1]._entity_for_mapper(prop.parent)
            parent = parent.parent[subclass_entity]
            if parent.parent:
                natural_parent = parent.parent[subclass_entity.mapper]
                self.is_unnatural = True
            else:
                natural_parent = parent
        elif natural_parent.parent and insp.is_aliased_class and (prop.parent is not insp.mapper) and insp.mapper.isa(prop.parent):
            natural_parent = parent.parent[prop.parent]
        self.prop = prop
        self.parent = parent
        self.path = parent.path + (prop,)
        self.natural_path = natural_parent.natural_path + (prop,)
        self.has_entity = prop._links_to_entity
        if prop._is_relationship:
            if TYPE_CHECKING:
                assert isinstance(prop, RelationshipProperty)
            self.entity = prop.entity
            self.mapper = prop.mapper
        else:
            self.entity = None
            self.mapper = None
        self._wildcard_path_loader_key = ('loader', parent.natural_path + self.prop._wildcard_token)
        self._default_path_loader_key = self.prop._default_path_loader_key
        self._loader_key = ('loader', self.natural_path)

    def _truncate_recursive(self) -> PropRegistry:
        earliest = None
        for i, token in enumerate(reversed(self.path[:-1])):
            if token is self.prop:
                earliest = i
        if earliest is None:
            return self
        else:
            return self.coerce(self.path[0:-(earliest + 1)])

    @property
    def entity_path(self) -> AbstractEntityRegistry:
        assert self.entity is not None
        return self[self.entity]

    def _getitem(self, entity: Union[int, slice, _InternalEntityType[Any]]) -> Union[AbstractEntityRegistry, _PathElementType, _PathRepresentation]:
        if isinstance(entity, (int, slice)):
            return self.path[entity]
        else:
            return SlotsEntityRegistry(self, entity)
    if not TYPE_CHECKING:
        __getitem__ = _getitem