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
class _SelectIsNotFrom(RoleImpl):
    __slots__ = ()

    def _raise_for_expected(self, element: Any, argname: Optional[str]=None, resolved: Optional[Any]=None, advice: Optional[str]=None, code: Optional[str]=None, err: Optional[Exception]=None, **kw: Any) -> NoReturn:
        if not advice and isinstance(element, roles.SelectStatementRole) or isinstance(resolved, roles.SelectStatementRole):
            advice = 'To create a FROM clause from a %s object, use the .subquery() method.' % (resolved.__class__ if resolved is not None else element,)
            code = '89ve'
        else:
            code = None
        super()._raise_for_expected(element, argname=argname, resolved=resolved, advice=advice, code=code, err=err, **kw)
        assert False