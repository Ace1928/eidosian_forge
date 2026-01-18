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
class AbstractEntityRegistry(CreatesToken):
    __slots__ = ('key', 'parent', 'is_aliased_class', 'path', 'entity', 'natural_path')
    has_entity = True
    is_entity = True
    parent: Union[RootRegistry, PropRegistry]
    key: _InternalEntityType[Any]
    entity: _InternalEntityType[Any]
    is_aliased_class: bool

    def __init__(self, parent: Union[RootRegistry, PropRegistry], entity: _InternalEntityType[Any]):
        self.key = entity
        self.parent = parent
        self.is_aliased_class = entity.is_aliased_class
        self.entity = entity
        self.path = parent.path + (entity,)
        if parent.path and (self.is_aliased_class or parent.is_unnatural):
            if entity.mapper.isa(parent.natural_path[-1].mapper):
                self.natural_path = parent.natural_path + (entity.mapper,)
            else:
                self.natural_path = parent.natural_path + (parent.natural_path[-1].entity,)
        else:
            self.natural_path = self.path

    def _truncate_recursive(self) -> AbstractEntityRegistry:
        return self.parent._truncate_recursive()[self.entity]

    @property
    def root_entity(self) -> _InternalEntityType[Any]:
        return self.odd_element(0)

    @property
    def entity_path(self) -> PathRegistry:
        return self

    @property
    def mapper(self) -> Mapper[Any]:
        return self.entity.mapper

    def __bool__(self) -> bool:
        return True

    def _getitem(self, entity: Any) -> Union[_PathElementType, _PathRepresentation, PathRegistry]:
        if isinstance(entity, (int, slice)):
            return self.path[entity]
        elif entity in PathToken._intern:
            return TokenRegistry(self, PathToken._intern[entity])
        else:
            return PropRegistry(self, entity)
    if not TYPE_CHECKING:
        __getitem__ = _getitem