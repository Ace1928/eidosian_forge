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
class ExpressionElementImpl(_ColumnCoercions, RoleImpl):
    __slots__ = ()

    def _literal_coercion(self, element, name=None, type_=None, argname=None, is_crud=False, **kw):
        if element is None and (not is_crud) and (type_ is None or not type_.should_evaluate_none):
            return elements.Null()
        else:
            try:
                return elements.BindParameter(name, element, type_, unique=True, _is_crud=is_crud)
            except exc.ArgumentError as err:
                self._raise_for_expected(element, err=err)

    def _raise_for_expected(self, element, argname=None, resolved=None, **kw):
        if isinstance(element, selectable.Values):
            advice = 'To create a column expression from a VALUES clause, use the .scalar_values() method.'
        elif isinstance(element, roles.AnonymizedFromClauseRole):
            advice = 'To create a column expression from a FROM clause row as a whole, use the .table_valued() method.'
        else:
            advice = None
        return super()._raise_for_expected(element, argname=argname, resolved=resolved, advice=advice, **kw)