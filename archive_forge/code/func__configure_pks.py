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
def _configure_pks(self) -> None:
    self.tables = sql_util.find_tables(self.persist_selectable)
    self._all_tables.update((t for t in self.tables))
    self._pks_by_table = {}
    self._cols_by_table = {}
    all_cols = util.column_set(chain(*[col.proxy_set for col in self._columntoproperty]))
    pk_cols = util.column_set((c for c in all_cols if c.primary_key))
    for fc in set(self.tables).union([self.persist_selectable]):
        if fc.primary_key and pk_cols.issuperset(fc.primary_key):
            self._pks_by_table[fc] = util.ordered_column_set(fc.primary_key).intersection(pk_cols)
        self._cols_by_table[fc] = util.ordered_column_set(fc.c).intersection(all_cols)
    if self._primary_key_argument:
        coerced_pk_arg = [self._str_arg_to_mapped_col('primary_key', c) if isinstance(c, str) else c for c in (coercions.expect(roles.DDLConstraintColumnRole, coerce_pk, argname='primary_key') for coerce_pk in self._primary_key_argument)]
    else:
        coerced_pk_arg = None
    if coerced_pk_arg:
        for k in coerced_pk_arg:
            if k.table not in self._pks_by_table:
                self._pks_by_table[k.table] = util.OrderedSet()
            self._pks_by_table[k.table].add(k)
    elif self.persist_selectable not in self._pks_by_table or len(self._pks_by_table[self.persist_selectable]) == 0:
        raise sa_exc.ArgumentError("Mapper %s could not assemble any primary key columns for mapped table '%s'" % (self, self.persist_selectable.description))
    elif self.local_table not in self._pks_by_table and isinstance(self.local_table, schema.Table):
        util.warn("Could not assemble any primary keys for locally mapped table '%s' - no rows will be persisted in this Table." % self.local_table.description)
    if self.inherits and (not self.concrete) and (not self._primary_key_argument):
        self.primary_key = self.inherits.primary_key
    else:
        primary_key: Collection[ColumnElement[Any]]
        if coerced_pk_arg:
            primary_key = [cc if cc is not None else c for cc, c in ((self.persist_selectable.corresponding_column(c), c) for c in coerced_pk_arg)]
        else:
            primary_key = sql_util.reduce_columns(self._pks_by_table[self.persist_selectable], ignore_nonexistent_tables=True)
        if len(primary_key) == 0:
            raise sa_exc.ArgumentError("Mapper %s could not assemble any primary key columns for mapped table '%s'" % (self, self.persist_selectable.description))
        self.primary_key = tuple(primary_key)
        self._log('Identified primary key columns: %s', primary_key)
    self._readonly_props = {self._columntoproperty[col] for col in self._columntoproperty if self._columntoproperty[col] not in self._identity_key_props and (not hasattr(col, 'table') or col.table not in self._cols_by_table)}