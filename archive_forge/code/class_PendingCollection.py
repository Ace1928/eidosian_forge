from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import weakref
from . import base
from . import exc as orm_exc
from . import interfaces
from ._typing import _O
from ._typing import is_collection_impl
from .base import ATTR_WAS_SET
from .base import INIT_OK
from .base import LoaderCallableStatus
from .base import NEVER_SET
from .base import NO_VALUE
from .base import PASSIVE_NO_INITIALIZE
from .base import PASSIVE_NO_RESULT
from .base import PASSIVE_OFF
from .base import SQL_OK
from .path_registry import PathRegistry
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
class PendingCollection:
    """A writable placeholder for an unloaded collection.

    Stores items appended to and removed from a collection that has not yet
    been loaded. When the collection is loaded, the changes stored in
    PendingCollection are applied to it to produce the final result.

    """
    __slots__ = ('deleted_items', 'added_items')
    deleted_items: util.IdentitySet
    added_items: util.OrderedIdentitySet

    def __init__(self) -> None:
        self.deleted_items = util.IdentitySet()
        self.added_items = util.OrderedIdentitySet()

    def merge_with_history(self, history: History) -> History:
        return history._merge(self.added_items, self.deleted_items)

    def append(self, value: Any) -> None:
        if value in self.deleted_items:
            self.deleted_items.remove(value)
        else:
            self.added_items.add(value)

    def remove(self, value: Any) -> None:
        if value in self.added_items:
            self.added_items.remove(value)
        else:
            self.deleted_items.add(value)