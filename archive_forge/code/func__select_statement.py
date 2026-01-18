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
def _select_statement(self, raw_columns, from_obj, where_criteria, having_criteria, label_style, order_by, for_update, hints, statement_hints, correlate, correlate_except, limit_clause, offset_clause, fetch_clause, fetch_clause_options, distinct, distinct_on, prefixes, suffixes, group_by, independent_ctes, independent_ctes_opts):
    statement = Select._create_raw_select(_raw_columns=raw_columns, _from_obj=from_obj, _label_style=label_style)
    if where_criteria:
        statement._where_criteria = where_criteria
    if having_criteria:
        statement._having_criteria = having_criteria
    if order_by:
        statement._order_by_clauses += tuple(order_by)
    if distinct_on:
        statement.distinct.non_generative(statement, *distinct_on)
    elif distinct:
        statement.distinct.non_generative(statement)
    if group_by:
        statement._group_by_clauses += tuple(group_by)
    statement._limit_clause = limit_clause
    statement._offset_clause = offset_clause
    statement._fetch_clause = fetch_clause
    statement._fetch_clause_options = fetch_clause_options
    statement._independent_ctes = independent_ctes
    statement._independent_ctes_opts = independent_ctes_opts
    if prefixes:
        statement._prefixes = prefixes
    if suffixes:
        statement._suffixes = suffixes
    statement._for_update_arg = for_update
    if hints:
        statement._hints = hints
    if statement_hints:
        statement._statement_hints = statement_hints
    if correlate:
        statement.correlate.non_generative(statement, *correlate)
    if correlate_except is not None:
        statement.correlate_except.non_generative(statement, *correlate_except)
    return statement