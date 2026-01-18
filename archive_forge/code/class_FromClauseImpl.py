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
class FromClauseImpl(_SelectIsNotFrom, _NoTextCoercion, RoleImpl):
    __slots__ = ()

    def _implicit_coercions(self, element: Any, resolved: Any, argname: Optional[str]=None, explicit_subquery: bool=False, allow_select: bool=True, **kw: Any) -> Any:
        if resolved._is_select_base:
            if explicit_subquery:
                return resolved.subquery()
            elif allow_select:
                util.warn_deprecated('Implicit coercion of SELECT and textual SELECT constructs into FROM clauses is deprecated; please call .subquery() on any Core select or ORM Query object in order to produce a subquery object.', version='1.4')
                return resolved._implicit_subquery
        elif resolved._is_text_clause:
            return resolved
        else:
            self._raise_for_expected(element, argname, resolved)

    def _post_coercion(self, element, deannotate=False, **kw):
        if deannotate:
            return element._deannotate()
        else:
            return element