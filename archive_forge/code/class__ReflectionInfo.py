from __future__ import annotations
import contextlib
from dataclasses import dataclass
from enum import auto
from enum import Flag
from enum import unique
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .base import Connection
from .base import Engine
from .. import exc
from .. import inspection
from .. import sql
from .. import util
from ..sql import operators
from ..sql import schema as sa_schema
from ..sql.cache_key import _ad_hoc_cache_key_from_args
from ..sql.elements import TextClause
from ..sql.type_api import TypeEngine
from ..sql.visitors import InternalTraversal
from ..util import topological
from ..util.typing import final
@dataclass
class _ReflectionInfo:
    columns: Dict[TableKey, List[ReflectedColumn]]
    pk_constraint: Dict[TableKey, Optional[ReflectedPrimaryKeyConstraint]]
    foreign_keys: Dict[TableKey, List[ReflectedForeignKeyConstraint]]
    indexes: Dict[TableKey, List[ReflectedIndex]]
    unique_constraints: Dict[TableKey, List[ReflectedUniqueConstraint]]
    table_comment: Dict[TableKey, Optional[ReflectedTableComment]]
    check_constraints: Dict[TableKey, List[ReflectedCheckConstraint]]
    table_options: Dict[TableKey, Dict[str, Any]]
    unreflectable: Dict[TableKey, exc.UnreflectableTableError]

    def update(self, other: _ReflectionInfo) -> None:
        for k, v in self.__dict__.items():
            ov = getattr(other, k)
            if ov is not None:
                if v is None:
                    setattr(self, k, ov)
                else:
                    v.update(ov)