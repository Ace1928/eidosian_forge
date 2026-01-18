from __future__ import annotations
import collections.abc as collections_abc
import numbers
import re
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import operators
from . import roles
from . import visitors
from ._typing import is_from_clause
from .base import ExecutableOption
from .base import Options
from .cache_key import HasCacheKey
from .visitors import Visitable
from .. import exc
from .. import inspection
from .. import util
from ..util.typing import Literal
class _ColumnCoercions(RoleImpl):
    __slots__ = ()

    def _warn_for_scalar_subquery_coercion(self):
        util.warn('implicitly coercing SELECT object to scalar subquery; please use the .scalar_subquery() method to produce a scalar subquery.')

    def _implicit_coercions(self, element, resolved, argname=None, **kw):
        original_element = element
        if not getattr(resolved, 'is_clause_element', False):
            self._raise_for_expected(original_element, argname, resolved)
        elif resolved._is_select_base:
            self._warn_for_scalar_subquery_coercion()
            return resolved.scalar_subquery()
        elif resolved._is_from_clause and isinstance(resolved, selectable.Subquery):
            self._warn_for_scalar_subquery_coercion()
            return resolved.element.scalar_subquery()
        elif self._role_class.allows_lambda and resolved._is_lambda_element:
            return resolved
        else:
            self._raise_for_expected(original_element, argname, resolved)