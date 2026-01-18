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
def _get_collection_history(self, state: InstanceState[Any], passive: PassiveFlag) -> WriteOnlyHistory[Any]:
    c: WriteOnlyHistory[Any]
    if self.key in state.committed_state:
        c = state.committed_state[self.key]
    else:
        c = self.collection_history_cls(self, state, PassiveFlag.PASSIVE_NO_FETCH)
    if state.has_identity and passive & PassiveFlag.INIT_OK:
        return self.collection_history_cls(self, state, passive, apply_to=c)
    else:
        return c