from __future__ import annotations
from decimal import Decimal
from enum import IntEnum
import itertools
import operator
import re
import typing
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple as typing_Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from ._typing import has_schema_attr
from ._typing import is_named_from_clause
from ._typing import is_quoted_name
from ._typing import is_tuple_type
from .annotation import Annotated
from .annotation import SupportsWrappingAnnotations
from .base import _clone
from .base import _expand_cloned
from .base import _generative
from .base import _NoArg
from .base import Executable
from .base import Generative
from .base import HasMemoized
from .base import Immutable
from .base import NO_ARG
from .base import SingletonConstant
from .cache_key import MemoizedHasCacheKey
from .cache_key import NO_CACHE
from .coercions import _document_text_coercion  # noqa
from .operators import ColumnOperators
from .traversals import HasCopyInternals
from .visitors import cloned_traverse
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .visitors import traverse
from .visitors import Visitable
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util import TypingOnly
from ..util.typing import Literal
from ..util.typing import Self
class ColumnClause(roles.DDLReferredColumnRole, roles.LabeledColumnExprRole[_T], roles.StrAsPlainColumnRole, Immutable, NamedColumn[_T]):
    """Represents a column expression from any textual string.

    The :class:`.ColumnClause`, a lightweight analogue to the
    :class:`_schema.Column` class, is typically invoked using the
    :func:`_expression.column` function, as in::

        from sqlalchemy import column

        id, name = column("id"), column("name")
        stmt = select(id, name).select_from("user")

    The above statement would produce SQL like::

        SELECT id, name FROM user

    :class:`.ColumnClause` is the immediate superclass of the schema-specific
    :class:`_schema.Column` object.  While the :class:`_schema.Column`
    class has all the
    same capabilities as :class:`.ColumnClause`, the :class:`.ColumnClause`
    class is usable by itself in those cases where behavioral requirements
    are limited to simple SQL expression generation.  The object has none of
    the associations with schema-level metadata or with execution-time
    behavior that :class:`_schema.Column` does,
    so in that sense is a "lightweight"
    version of :class:`_schema.Column`.

    Full details on :class:`.ColumnClause` usage is at
    :func:`_expression.column`.

    .. seealso::

        :func:`_expression.column`

        :class:`_schema.Column`

    """
    table: Optional[FromClause]
    is_literal: bool
    __visit_name__ = 'column'
    _traverse_internals: _TraverseInternalsType = [('name', InternalTraversal.dp_anon_name), ('type', InternalTraversal.dp_type), ('table', InternalTraversal.dp_clauseelement), ('is_literal', InternalTraversal.dp_boolean)]
    onupdate: Optional[DefaultGenerator] = None
    default: Optional[DefaultGenerator] = None
    server_default: Optional[FetchedValue] = None
    server_onupdate: Optional[FetchedValue] = None
    _is_multiparam_column = False

    @property
    def _is_star(self):
        return self.is_literal and self.name == '*'

    def __init__(self, text: str, type_: Optional[_TypeEngineArgument[_T]]=None, is_literal: bool=False, _selectable: Optional[FromClause]=None):
        self.key = self.name = text
        self.table = _selectable
        self.type = type_api.to_instance(type_)
        self.is_literal = is_literal

    def get_children(self, *, column_tables=False, **kw):
        return []

    @property
    def entity_namespace(self):
        if self.table is not None:
            return self.table.entity_namespace
        else:
            return super().entity_namespace

    def _clone(self, detect_subquery_cols=False, **kw):
        if detect_subquery_cols and self.table is not None and self.table._is_subquery:
            clone = kw.pop('clone')
            table = clone(self.table, **kw)
            new = table.c.corresponding_column(self)
            return new
        return super()._clone(**kw)

    @HasMemoized_ro_memoized_attribute
    def _from_objects(self) -> List[FromClause]:
        t = self.table
        if t is not None:
            return [t]
        else:
            return []

    @HasMemoized.memoized_attribute
    def _render_label_in_columns_clause(self):
        return self.table is not None

    @property
    def _ddl_label(self):
        return self._gen_tq_label(self.name, dedupe_on_key=False)

    def _compare_name_for_result(self, other):
        if self.is_literal or self.table is None or self.table._is_textual or (not hasattr(other, 'proxy_set')) or (isinstance(other, ColumnClause) and (other.is_literal or other.table is None or other.table._is_textual)):
            return hasattr(other, 'name') and self.name == other.name or (hasattr(other, '_tq_label') and self._tq_label == other._tq_label)
        else:
            return other.proxy_set.intersection(self.proxy_set)

    def _gen_tq_label(self, name: str, dedupe_on_key: bool=True) -> Optional[str]:
        """generate table-qualified label

        for a table-bound column this is <tablename>_<columnname>.

        used primarily for LABEL_STYLE_TABLENAME_PLUS_COL
        as well as the .columns collection on a Join object.

        """
        label: str
        t = self.table
        if self.is_literal:
            return None
        elif t is not None and is_named_from_clause(t):
            if has_schema_attr(t) and t.schema:
                label = t.schema.replace('.', '_') + '_' + t.name + '_' + name
            else:
                assert not TYPE_CHECKING or isinstance(t, NamedFromClause)
                label = t.name + '_' + name
            if is_quoted_name(name) and name.quote is not None:
                if is_quoted_name(label):
                    label.quote = name.quote
                else:
                    label = quoted_name(label, name.quote)
            elif is_quoted_name(t.name) and t.name.quote is not None:
                assert not isinstance(label, quoted_name)
                label = quoted_name(label, t.name.quote)
            if dedupe_on_key:
                if label in t.c:
                    _label = label
                    counter = 1
                    while _label in t.c:
                        _label = label + '_' + str(counter)
                        counter += 1
                    label = _label
            return coercions.expect(roles.TruncatedLabelRole, label)
        else:
            return name

    def _make_proxy(self, selectable: FromClause, *, name: Optional[str]=None, key: Optional[str]=None, name_is_truncatable: bool=False, compound_select_cols: Optional[Sequence[ColumnElement[Any]]]=None, disallow_is_literal: bool=False, **kw: Any) -> typing_Tuple[str, ColumnClause[_T]]:
        is_literal = not disallow_is_literal and self.is_literal and (name is None or name == self.name)
        c = self._constructor(coercions.expect(roles.TruncatedLabelRole, name or self.name) if name_is_truncatable else name or self.name, type_=self.type, _selectable=selectable, is_literal=is_literal)
        c._propagate_attrs = selectable._propagate_attrs
        if name is None:
            c.key = self.key
        if compound_select_cols:
            c._proxies = list(compound_select_cols)
        else:
            c._proxies = [self]
        if selectable._is_clone_of is not None:
            c._is_clone_of = selectable._is_clone_of.columns.get(c.key)
        return (c.key, c)