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
class SelectsRows(ReturnsRows):
    """Sub-base of ReturnsRows for elements that deliver rows
    directly, namely SELECT and INSERT/UPDATE/DELETE..RETURNING"""
    _label_style: SelectLabelStyle = LABEL_STYLE_NONE

    def _generate_columns_plus_names(self, anon_for_dupe_key: bool, cols: Optional[_SelectIterable]=None) -> List[_ColumnsPlusNames]:
        """Generate column names as rendered in a SELECT statement by
        the compiler.

        This is distinct from the _column_naming_convention generator that's
        intended for population of .c collections and similar, which has
        different rules.   the collection returned here calls upon the
        _column_naming_convention as well.

        """
        if cols is None:
            cols = self._all_selected_columns
        key_naming_convention = SelectState._column_naming_convention(self._label_style)
        names = {}
        result: List[_ColumnsPlusNames] = []
        result_append = result.append
        table_qualified = self._label_style is LABEL_STYLE_TABLENAME_PLUS_COL
        label_style_none = self._label_style is LABEL_STYLE_NONE
        dedupe_hash = 1
        for c in cols:
            repeated = False
            if not c._render_label_in_columns_clause:
                effective_name = required_label_name = fallback_label_name = None
            elif label_style_none:
                if TYPE_CHECKING:
                    assert is_column_element(c)
                effective_name = required_label_name = None
                fallback_label_name = c._non_anon_label or c._anon_name_label
            else:
                if TYPE_CHECKING:
                    assert is_column_element(c)
                if table_qualified:
                    required_label_name = effective_name = fallback_label_name = c._tq_label
                else:
                    effective_name = fallback_label_name = c._non_anon_label
                    required_label_name = None
                if effective_name is None:
                    expr_label = c._expression_label
                    if expr_label is None:
                        repeated = c._anon_name_label in names
                        names[c._anon_name_label] = c
                        effective_name = required_label_name = None
                        if repeated:
                            if table_qualified:
                                fallback_label_name = c._dedupe_anon_tq_label_idx(dedupe_hash)
                                dedupe_hash += 1
                            else:
                                fallback_label_name = c._dedupe_anon_label_idx(dedupe_hash)
                                dedupe_hash += 1
                        else:
                            fallback_label_name = c._anon_name_label
                    else:
                        required_label_name = effective_name = fallback_label_name = expr_label
            if effective_name is not None:
                if TYPE_CHECKING:
                    assert is_column_element(c)
                if effective_name in names:
                    if hash(names[effective_name]) != hash(c):
                        if table_qualified:
                            required_label_name = fallback_label_name = c._anon_tq_label
                        else:
                            required_label_name = fallback_label_name = c._anon_name_label
                        if anon_for_dupe_key and required_label_name in names:
                            assert hash(names[required_label_name]) == hash(c)
                            if table_qualified:
                                required_label_name = fallback_label_name = c._dedupe_anon_tq_label_idx(dedupe_hash)
                                dedupe_hash += 1
                            else:
                                required_label_name = fallback_label_name = c._dedupe_anon_label_idx(dedupe_hash)
                                dedupe_hash += 1
                            repeated = True
                        else:
                            names[required_label_name] = c
                    elif anon_for_dupe_key:
                        if table_qualified:
                            required_label_name = fallback_label_name = c._dedupe_anon_tq_label_idx(dedupe_hash)
                            dedupe_hash += 1
                        else:
                            required_label_name = fallback_label_name = c._dedupe_anon_label_idx(dedupe_hash)
                            dedupe_hash += 1
                        repeated = True
                else:
                    names[effective_name] = c
            result_append(_ColumnsPlusNames(required_label_name, key_naming_convention(c), fallback_label_name, c, repeated))
        return result