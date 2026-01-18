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
class WriteOnlyHistory(Generic[_T]):
    """Overrides AttributeHistory to receive append/remove events directly."""
    unchanged_items: util.OrderedIdentitySet
    added_items: util.OrderedIdentitySet
    deleted_items: util.OrderedIdentitySet
    _reconcile_collection: bool

    def __init__(self, attr: WriteOnlyAttributeImpl, state: InstanceState[_T], passive: PassiveFlag, apply_to: Optional[WriteOnlyHistory[_T]]=None) -> None:
        if apply_to:
            if passive & PassiveFlag.SQL_OK:
                raise exc.InvalidRequestError(f"Attribute {attr} can't load the existing state from the database for this operation; full iteration is not permitted.  If this is a delete operation, configure passive_deletes=True on the {attr} relationship in order to resolve this error.")
            self.unchanged_items = apply_to.unchanged_items
            self.added_items = apply_to.added_items
            self.deleted_items = apply_to.deleted_items
            self._reconcile_collection = apply_to._reconcile_collection
        else:
            self.deleted_items = util.OrderedIdentitySet()
            self.added_items = util.OrderedIdentitySet()
            self.unchanged_items = util.OrderedIdentitySet()
            self._reconcile_collection = False

    @property
    def added_plus_unchanged(self) -> List[_T]:
        return list(self.added_items.union(self.unchanged_items))

    @property
    def all_items(self) -> List[_T]:
        return list(self.added_items.union(self.unchanged_items).union(self.deleted_items))

    def as_history(self) -> attributes.History:
        if self._reconcile_collection:
            added = self.added_items.difference(self.unchanged_items)
            deleted = self.deleted_items.intersection(self.unchanged_items)
            unchanged = self.unchanged_items.difference(deleted)
        else:
            added, unchanged, deleted = (self.added_items, self.unchanged_items, self.deleted_items)
        return attributes.History(list(added), list(unchanged), list(deleted))

    def indexed(self, index: Union[int, slice]) -> Union[List[_T], _T]:
        return list(self.added_items)[index]

    def add_added(self, value: _T) -> None:
        self.added_items.add(value)

    def add_removed(self, value: _T) -> None:
        if value in self.added_items:
            self.added_items.remove(value)
        else:
            self.deleted_items.add(value)