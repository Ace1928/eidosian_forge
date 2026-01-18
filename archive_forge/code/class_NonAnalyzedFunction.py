from __future__ import annotations
import collections.abc as collections_abc
import inspect
import itertools
import operator
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import cache_key as _cache_key
from . import coercions
from . import elements
from . import roles
from . import schema
from . import visitors
from .base import _clone
from .base import Executable
from .base import Options
from .cache_key import CacheConst
from .operators import ColumnOperators
from .. import exc
from .. import inspection
from .. import util
from ..util.typing import Literal
class NonAnalyzedFunction:
    __slots__ = ('expr',)
    closure_bindparams: Optional[List[BindParameter[Any]]] = None
    bindparam_trackers: Optional[List[_BoundParameterGetter]] = None
    is_sequence = False
    expr: ClauseElement

    def __init__(self, expr: ClauseElement):
        self.expr = expr

    @property
    def expected_expr(self) -> ClauseElement:
        return self.expr