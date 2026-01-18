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
class OnClauseImpl(_ColumnCoercions, RoleImpl):
    __slots__ = ()
    _coerce_consts = True

    def _literal_coercion(self, element, name=None, type_=None, argname=None, is_crud=False, **kw):
        self._raise_for_expected(element)

    def _post_coercion(self, resolved, original_element=None, **kw):
        if isinstance(original_element, roles.JoinTargetRole):
            return original_element
        return resolved