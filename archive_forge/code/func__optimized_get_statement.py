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
def _optimized_get_statement(self, state, attribute_names):
    """assemble a WHERE clause which retrieves a given state by primary
        key, using a minimized set of tables.

        Applies to a joined-table inheritance mapper where the
        requested attribute names are only present on joined tables,
        not the base table.  The WHERE clause attempts to include
        only those tables to minimize joins.

        """
    props = self._props
    col_attribute_names = set(attribute_names).intersection(state.mapper.column_attrs.keys())
    tables: Set[FromClause] = set(chain(*[sql_util.find_tables(c, check_columns=True) for key in col_attribute_names for c in props[key].columns]))
    if self.base_mapper.local_table in tables:
        return None

    def visit_binary(binary):
        leftcol = binary.left
        rightcol = binary.right
        if leftcol is None or rightcol is None:
            return
        if leftcol.table not in tables:
            leftval = self._get_committed_state_attr_by_column(state, state.dict, leftcol, passive=PassiveFlag.PASSIVE_NO_INITIALIZE)
            if leftval in orm_util._none_set:
                raise _OptGetColumnsNotAvailable()
            binary.left = sql.bindparam(None, leftval, type_=binary.right.type)
        elif rightcol.table not in tables:
            rightval = self._get_committed_state_attr_by_column(state, state.dict, rightcol, passive=PassiveFlag.PASSIVE_NO_INITIALIZE)
            if rightval in orm_util._none_set:
                raise _OptGetColumnsNotAvailable()
            binary.right = sql.bindparam(None, rightval, type_=binary.right.type)
    allconds: List[ColumnElement[bool]] = []
    start = False
    for mapper in reversed(list(self.iterate_to_root())):
        if mapper.local_table in tables:
            start = True
        elif not isinstance(mapper.local_table, expression.TableClause):
            return None
        if start and (not mapper.single):
            assert mapper.inherits
            assert not mapper.concrete
            assert mapper.inherit_condition is not None
            allconds.append(mapper.inherit_condition)
            tables.add(mapper.local_table)
    try:
        _traversed = visitors.cloned_traverse(allconds[0], {}, {'binary': visit_binary})
    except _OptGetColumnsNotAvailable:
        return None
    else:
        allconds[0] = _traversed
    cond = sql.and_(*allconds)
    cols = []
    for key in col_attribute_names:
        cols.extend(props[key].columns)
    return sql.select(*cols).where(cond).set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)