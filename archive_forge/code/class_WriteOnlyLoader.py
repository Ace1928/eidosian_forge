from __future__ import annotations
from typing import Any
from typing import Collection
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.sql import bindparam
from . import attributes
from . import interfaces
from . import relationships
from . import strategies
from .base import NEVER_SET
from .base import object_mapper
from .base import PassiveFlag
from .base import RelationshipDirection
from .. import exc
from .. import inspect
from .. import log
from .. import util
from ..sql import delete
from ..sql import insert
from ..sql import select
from ..sql import update
from ..sql.dml import Delete
from ..sql.dml import Insert
from ..sql.dml import Update
from ..util.typing import Literal
@log.class_logger
@relationships.RelationshipProperty.strategy_for(lazy='write_only')
class WriteOnlyLoader(strategies.AbstractRelationshipLoader, log.Identified):
    impl_class = WriteOnlyAttributeImpl

    def init_class_attribute(self, mapper: Mapper[Any]) -> None:
        self.is_class_level = True
        if not self.uselist or self.parent_property.direction not in (interfaces.ONETOMANY, interfaces.MANYTOMANY):
            raise exc.InvalidRequestError("On relationship %s, 'dynamic' loaders cannot be used with many-to-one/one-to-one relationships and/or uselist=False." % self.parent_property)
        strategies._register_attribute(self.parent_property, mapper, useobject=True, impl_class=self.impl_class, target_mapper=self.parent_property.mapper, order_by=self.parent_property.order_by, query_class=self.parent_property.query_class)