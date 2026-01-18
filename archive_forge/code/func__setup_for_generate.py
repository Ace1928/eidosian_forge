from __future__ import annotations
import itertools
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import interfaces
from . import loading
from .base import _is_aliased_class
from .interfaces import ORMColumnDescription
from .interfaces import ORMColumnsClauseRole
from .path_registry import PathRegistry
from .util import _entity_corresponds_to
from .util import _ORMJoin
from .util import _TraceAdaptRole
from .util import AliasedClass
from .util import Bundle
from .util import ORMAdapter
from .util import ORMStatementAdapter
from .. import exc as sa_exc
from .. import future
from .. import inspect
from .. import sql
from .. import util
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import _TP
from ..sql._typing import is_dml
from ..sql._typing import is_insert_update
from ..sql._typing import is_select_base
from ..sql.base import _select_iterables
from ..sql.base import CacheableOptions
from ..sql.base import CompileState
from ..sql.base import Executable
from ..sql.base import Generative
from ..sql.base import Options
from ..sql.dml import UpdateBase
from ..sql.elements import GroupedElement
from ..sql.elements import TextClause
from ..sql.selectable import CompoundSelectState
from ..sql.selectable import LABEL_STYLE_DISAMBIGUATE_ONLY
from ..sql.selectable import LABEL_STYLE_NONE
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import Select
from ..sql.selectable import SelectLabelStyle
from ..sql.selectable import SelectState
from ..sql.selectable import TypedReturnsRows
from ..sql.visitors import InternalTraversal
def _setup_for_generate(self):
    query = self.select_statement
    self.statement = None
    self._join_entities = ()
    if self.compile_options._set_base_alias:
        self._set_select_from_alias()
    for memoized_entities in query._memoized_select_entities:
        if memoized_entities._setup_joins:
            self._join(memoized_entities._setup_joins, self._memoized_entities[memoized_entities])
    if query._setup_joins:
        self._join(query._setup_joins, self._entities)
    current_adapter = self._get_current_adapter()
    if query._where_criteria:
        self._where_criteria = query._where_criteria
        if current_adapter:
            self._where_criteria = tuple((current_adapter(crit, True) for crit in self._where_criteria))
    self.order_by = self._adapt_col_list(query._order_by_clauses, current_adapter) if current_adapter and query._order_by_clauses not in (None, False) else query._order_by_clauses
    if query._having_criteria:
        self._having_criteria = tuple((current_adapter(crit, True) if current_adapter else crit for crit in query._having_criteria))
    self.group_by = self._adapt_col_list(util.flatten_iterator(query._group_by_clauses), current_adapter) if current_adapter and query._group_by_clauses not in (None, False) else query._group_by_clauses or None
    if self.eager_order_by:
        adapter = self.from_clauses[0]._target_adapter
        self.eager_order_by = adapter.copy_and_process(self.eager_order_by)
    if query._distinct_on:
        self.distinct_on = self._adapt_col_list(query._distinct_on, current_adapter)
    else:
        self.distinct_on = ()
    self.distinct = query._distinct
    if query._correlate:
        self.correlate = tuple(util.flatten_iterator((sql_util.surface_selectables(s) if s is not None else None for s in query._correlate)))
    elif query._correlate_except is not None:
        self.correlate_except = tuple(util.flatten_iterator((sql_util.surface_selectables(s) if s is not None else None for s in query._correlate_except)))
    elif not query._auto_correlate:
        self.correlate = (None,)
    self._for_update_arg = query._for_update_arg
    if self.compile_options._is_star and len(self._entities) != 1:
        raise sa_exc.CompileError("Can't generate ORM query that includes multiple expressions at the same time as '*'; query for '*' alone if present")
    for entity in self._entities:
        entity.setup_compile_state(self)
    for rec in self.create_eager_joins:
        strategy = rec[0]
        strategy(self, *rec[1:])
    if self.compile_options._enable_single_crit:
        self._adjust_for_extra_criteria()
    if not self.primary_columns:
        if self.compile_options._only_load_props:
            assert False, 'no columns were included in _only_load_props'
        raise sa_exc.InvalidRequestError('Query contains no columns with which to SELECT from.')
    if not self.from_clauses:
        self.from_clauses = list(self._fallback_from_clauses)
    if self.order_by is False:
        self.order_by = None
    if self.multi_row_eager_loaders and self.eager_adding_joins and self._should_nest_selectable:
        self.statement = self._compound_eager_statement()
    else:
        self.statement = self._simple_statement()
    if self.for_statement:
        ezero = self._mapper_zero()
        if ezero is not None:
            self.statement = self.statement._annotate({'deepentity': ezero})