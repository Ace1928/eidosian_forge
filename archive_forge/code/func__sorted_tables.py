from __future__ import annotations
from collections import deque
from functools import reduce
from itertools import chain
import sys
import threading
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Deque
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import exc as orm_exc
from . import instrumentation
from . import loading
from . import properties
from . import util as orm_util
from ._typing import _O
from .base import _class_to_mapper
from .base import _parse_mapper_argument
from .base import _state_mapper
from .base import PassiveFlag
from .base import state_str
from .interfaces import _MappedAttribute
from .interfaces import EXT_SKIP
from .interfaces import InspectionAttr
from .interfaces import MapperProperty
from .interfaces import ORMEntityColumnsClauseRole
from .interfaces import ORMFromClauseRole
from .interfaces import StrategizedProperty
from .path_registry import PathRegistry
from .. import event
from .. import exc as sa_exc
from .. import inspection
from .. import log
from .. import schema
from .. import sql
from .. import util
from ..event import dispatcher
from ..event import EventTarget
from ..sql import base as sql_base
from ..sql import coercions
from ..sql import expression
from ..sql import operators
from ..sql import roles
from ..sql import TableClause
from ..sql import util as sql_util
from ..sql import visitors
from ..sql.cache_key import MemoizedHasCacheKey
from ..sql.elements import KeyedColumnElement
from ..sql.schema import Column
from ..sql.schema import Table
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import HasMemoized
from ..util import HasMemoized_ro_memoized_attribute
from ..util.typing import Literal
@HasMemoized.memoized_attribute
def _sorted_tables(self):
    table_to_mapper: Dict[TableClause, Mapper[Any]] = {}
    for mapper in self.base_mapper.self_and_descendants:
        for t in mapper.tables:
            table_to_mapper.setdefault(t, mapper)
    extra_dependencies = []
    for table, mapper in table_to_mapper.items():
        super_ = mapper.inherits
        if super_:
            extra_dependencies.extend([(super_table, table) for super_table in super_.tables])

    def skip(fk):
        parent = table_to_mapper.get(fk.parent.table)
        dep = table_to_mapper.get(fk.column.table)
        if parent is not None and dep is not None and (dep is not parent) and (dep.inherit_condition is not None):
            cols = set(sql_util._find_columns(dep.inherit_condition))
            if parent.inherit_condition is not None:
                cols = cols.union(sql_util._find_columns(parent.inherit_condition))
                return fk.parent not in cols and fk.column not in cols
            else:
                return fk.parent not in cols
        return False
    sorted_ = sql_util.sort_tables(table_to_mapper, skip_fn=skip, extra_dependencies=extra_dependencies)
    ret = util.OrderedDict()
    for t in sorted_:
        ret[t] = table_to_mapper[t]
    return ret