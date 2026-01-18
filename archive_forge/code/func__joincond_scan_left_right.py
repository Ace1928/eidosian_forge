from __future__ import annotations
import collections
from enum import Enum
import itertools
from typing import AbstractSet
from typing import Any as TODO_Any
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import cache_key
from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from . import visitors
from ._typing import _ColumnsClauseArgument
from ._typing import _no_kw
from ._typing import _TP
from ._typing import is_column_element
from ._typing import is_select_statement
from ._typing import is_subquery
from ._typing import is_table
from ._typing import is_text_clause
from .annotation import Annotated
from .annotation import SupportsCloneAnnotations
from .base import _clone
from .base import _cloned_difference
from .base import _cloned_intersection
from .base import _entity_namespace_key
from .base import _EntityNamespace
from .base import _expand_cloned
from .base import _from_objects
from .base import _generative
from .base import _never_select_column
from .base import _NoArg
from .base import _select_iterables
from .base import CacheableOptions
from .base import ColumnCollection
from .base import ColumnSet
from .base import CompileState
from .base import DedupeColumnCollection
from .base import Executable
from .base import Generative
from .base import HasCompileState
from .base import HasMemoized
from .base import Immutable
from .coercions import _document_text_coercion
from .elements import _anonymous_label
from .elements import BindParameter
from .elements import BooleanClauseList
from .elements import ClauseElement
from .elements import ClauseList
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import DQLDMLClauseElement
from .elements import GroupedElement
from .elements import literal_column
from .elements import TableValuedColumn
from .elements import UnaryExpression
from .operators import OperatorType
from .sqltypes import NULLTYPE
from .visitors import _TraverseInternalsType
from .visitors import InternalTraversal
from .visitors import prefix_anon_map
from .. import exc
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
@classmethod
@util.preload_module('sqlalchemy.sql.util')
def _joincond_scan_left_right(cls, a: FromClause, a_subset: Optional[FromClause], b: FromClause, consider_as_foreign_keys: Optional[AbstractSet[ColumnClause[Any]]]) -> collections.defaultdict[Optional[ForeignKeyConstraint], List[Tuple[ColumnClause[Any], ColumnClause[Any]]]]:
    sql_util = util.preloaded.sql_util
    a = coercions.expect(roles.FromClauseRole, a)
    b = coercions.expect(roles.FromClauseRole, b)
    constraints: collections.defaultdict[Optional[ForeignKeyConstraint], List[Tuple[ColumnClause[Any], ColumnClause[Any]]]] = collections.defaultdict(list)
    for left in (a_subset, a):
        if left is None:
            continue
        for fk in sorted(b.foreign_keys, key=lambda fk: fk.parent._creation_order):
            if consider_as_foreign_keys is not None and fk.parent not in consider_as_foreign_keys:
                continue
            try:
                col = fk.get_referent(left)
            except exc.NoReferenceError as nrte:
                table_names = {t.name for t in sql_util.find_tables(left)}
                if nrte.table_name in table_names:
                    raise
                else:
                    continue
            if col is not None:
                constraints[fk.constraint].append((col, fk.parent))
        if left is not b:
            for fk in sorted(left.foreign_keys, key=lambda fk: fk.parent._creation_order):
                if consider_as_foreign_keys is not None and fk.parent not in consider_as_foreign_keys:
                    continue
                try:
                    col = fk.get_referent(b)
                except exc.NoReferenceError as nrte:
                    table_names = {t.name for t in sql_util.find_tables(b)}
                    if nrte.table_name in table_names:
                        raise
                    else:
                        continue
                if col is not None:
                    constraints[fk.constraint].append((col, fk.parent))
        if constraints:
            break
    return constraints