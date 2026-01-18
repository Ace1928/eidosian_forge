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
def _reflect_check_constraints(self, _reflect_info: _ReflectionInfo, table_key: TableKey, table: sa_schema.Table, cols_by_orig_name: Dict[str, sa_schema.Column[Any]], include_columns: Optional[Collection[str]], exclude_columns: Collection[str], reflection_options: Dict[str, Any]) -> None:
    constraints = _reflect_info.check_constraints.get(table_key, [])
    for const_d in constraints:
        table.append_constraint(sa_schema.CheckConstraint(**const_d))