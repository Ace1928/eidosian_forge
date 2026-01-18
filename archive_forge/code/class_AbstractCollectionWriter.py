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
class AbstractCollectionWriter(Generic[_T]):
    """Virtual collection which includes append/remove methods that synchronize
    into the attribute event system.

    """
    if not TYPE_CHECKING:
        __slots__ = ()
    instance: _T
    _from_obj: Tuple[FromClause, ...]

    def __init__(self, attr: WriteOnlyAttributeImpl, state: InstanceState[_T]):
        instance = state.obj()
        if TYPE_CHECKING:
            assert instance
        self.instance = instance
        self.attr = attr
        mapper = object_mapper(instance)
        prop = mapper._props[self.attr.key]
        if prop.secondary is not None:
            self._from_obj = (prop.mapper.__clause_element__(), prop.secondary)
        else:
            self._from_obj = ()
        self._where_criteria = (prop._with_parent(instance, alias_secondary=False),)
        if self.attr.order_by:
            self._order_by_clauses = self.attr.order_by
        else:
            self._order_by_clauses = ()

    def _add_all_impl(self, iterator: Iterable[_T]) -> None:
        for item in iterator:
            self.attr.append(attributes.instance_state(self.instance), attributes.instance_dict(self.instance), item, None)

    def _remove_impl(self, item: _T) -> None:
        self.attr.remove(attributes.instance_state(self.instance), attributes.instance_dict(self.instance), item, None)