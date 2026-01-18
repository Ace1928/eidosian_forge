from __future__ import annotations
from abc import abstractmethod
import re
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.types import NULLTYPE
from . import schemaobj
from .base import BatchOperations
from .base import Operations
from .. import util
from ..util import sqla_compat
class ModifyTableOps(OpContainer):
    """Contains a sequence of operations that all apply to a single Table."""

    def __init__(self, table_name: str, ops: Sequence[MigrateOperation], *, schema: Optional[str]=None) -> None:
        super().__init__(ops)
        self.table_name = table_name
        self.schema = schema

    def reverse(self) -> ModifyTableOps:
        return ModifyTableOps(self.table_name, ops=list(reversed([op.reverse() for op in self.ops])), schema=self.schema)