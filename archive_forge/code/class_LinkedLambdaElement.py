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
class LinkedLambdaElement(StatementLambdaElement):
    """Represent subsequent links of a :class:`.StatementLambdaElement`."""
    parent_lambda: StatementLambdaElement

    def __init__(self, fn: _StmtLambdaElementType[Any], parent_lambda: StatementLambdaElement, opts: Union[Type[LambdaOptions], LambdaOptions]):
        self.opts = opts
        self.fn = fn
        self.parent_lambda = parent_lambda
        self.tracker_key = parent_lambda.tracker_key + (fn.__code__,)
        self._retrieve_tracker_rec(fn, self, opts)
        self._propagate_attrs = parent_lambda._propagate_attrs

    def _invoke_user_fn(self, fn, *arg):
        return fn(self.parent_lambda._resolved)