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
class RoleImpl:
    __slots__ = ('_role_class', 'name', '_use_inspection')

    def _literal_coercion(self, element, **kw):
        raise NotImplementedError()
    _post_coercion: Any = None
    _resolve_literal_only = False
    _skip_clauseelement_for_target_match = False

    def __init__(self, role_class):
        self._role_class = role_class
        self.name = role_class._role_name
        self._use_inspection = issubclass(role_class, roles.UsesInspection)

    def _implicit_coercions(self, element: Any, resolved: Any, argname: Optional[str]=None, **kw: Any) -> Any:
        self._raise_for_expected(element, argname, resolved)

    def _raise_for_expected(self, element: Any, argname: Optional[str]=None, resolved: Optional[Any]=None, advice: Optional[str]=None, code: Optional[str]=None, err: Optional[Exception]=None, **kw: Any) -> NoReturn:
        if resolved is not None and resolved is not element:
            got = '%r object resolved from %r object' % (resolved, element)
        else:
            got = repr(element)
        if argname:
            msg = '%s expected for argument %r; got %s.' % (self.name, argname, got)
        else:
            msg = '%s expected, got %s.' % (self.name, got)
        if advice:
            msg += ' ' + advice
        raise exc.ArgumentError(msg, code=code) from err