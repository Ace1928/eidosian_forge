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
class NamedColumn(KeyedColumnElement[_T]):
    is_literal = False
    table: Optional[FromClause] = None
    name: str
    key: str

    def _compare_name_for_result(self, other):
        return hasattr(other, 'name') and self.name == other.name or (hasattr(other, '_label') and self._label == other._label)

    @util.ro_memoized_property
    def description(self) -> str:
        return self.name

    @HasMemoized.memoized_attribute
    def _tq_key_label(self):
        """table qualified label based on column key.

        for table-bound columns this is <tablename>_<column key/proxy key>;

        all other expressions it resolves to key/proxy key.

        """
        proxy_key = self._proxy_key
        if proxy_key and proxy_key != self.name:
            return self._gen_tq_label(proxy_key)
        else:
            return self._tq_label

    @HasMemoized.memoized_attribute
    def _tq_label(self) -> Optional[str]:
        """table qualified label based on column name.

        for table-bound columns this is <tablename>_<columnname>; all other
        expressions it resolves to .name.

        """
        return self._gen_tq_label(self.name)

    @HasMemoized.memoized_attribute
    def _render_label_in_columns_clause(self):
        return True

    @HasMemoized.memoized_attribute
    def _non_anon_label(self):
        return self.name

    def _gen_tq_label(self, name: str, dedupe_on_key: bool=True) -> Optional[str]:
        return name

    def _bind_param(self, operator: OperatorType, obj: Any, type_: Optional[TypeEngine[_T]]=None, expanding: bool=False) -> BindParameter[_T]:
        return BindParameter(self.key, obj, _compared_to_operator=operator, _compared_to_type=self.type, type_=type_, unique=True, expanding=expanding)

    def _make_proxy(self, selectable: FromClause, *, name: Optional[str]=None, key: Optional[str]=None, name_is_truncatable: bool=False, compound_select_cols: Optional[Sequence[ColumnElement[Any]]]=None, disallow_is_literal: bool=False, **kw: Any) -> typing_Tuple[str, ColumnClause[_T]]:
        c = ColumnClause(coercions.expect(roles.TruncatedLabelRole, name or self.name) if name_is_truncatable else name or self.name, type_=self.type, _selectable=selectable, is_literal=False)
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