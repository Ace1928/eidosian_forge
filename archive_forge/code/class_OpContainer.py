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
class OpContainer(MigrateOperation):
    """Represent a sequence of operations operation."""

    def __init__(self, ops: Sequence[MigrateOperation]=()) -> None:
        self.ops = list(ops)

    def is_empty(self) -> bool:
        return not self.ops

    def as_diffs(self) -> Any:
        return list(OpContainer._ops_as_diffs(self))

    @classmethod
    def _ops_as_diffs(cls, migrations: OpContainer) -> Iterator[Tuple[Any, ...]]:
        for op in migrations.ops:
            if hasattr(op, 'ops'):
                yield from cls._ops_as_diffs(cast('OpContainer', op))
            else:
                yield op.to_diff_tuple()