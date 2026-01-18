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
class JoinTargetImpl(RoleImpl):
    __slots__ = ()
    _skip_clauseelement_for_target_match = True

    def _literal_coercion(self, element, argname=None, **kw):
        self._raise_for_expected(element, argname)

    def _implicit_coercions(self, element: Any, resolved: Any, argname: Optional[str]=None, legacy: bool=False, **kw: Any) -> Any:
        if isinstance(element, roles.JoinTargetRole):
            return element
        elif legacy and resolved._is_select_base:
            util.warn_deprecated('Implicit coercion of SELECT and textual SELECT constructs into FROM clauses is deprecated; please call .subquery() on any Core select or ORM Query object in order to produce a subquery object.', version='1.4')
            return resolved
        else:
            self._raise_for_expected(element, argname, resolved)