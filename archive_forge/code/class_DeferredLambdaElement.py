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
class DeferredLambdaElement(LambdaElement):
    """A LambdaElement where the lambda accepts arguments and is
    invoked within the compile phase with special context.

    This lambda doesn't normally produce its real SQL expression outside of the
    compile phase.  It is passed a fixed set of initial arguments
    so that it can generate a sample expression.

    """

    def __init__(self, fn: _AnyLambdaType, role: Type[roles.SQLRole], opts: Union[Type[LambdaOptions], LambdaOptions]=LambdaOptions, lambda_args: Tuple[Any, ...]=()):
        self.lambda_args = lambda_args
        super().__init__(fn, role, opts)

    def _invoke_user_fn(self, fn, *arg):
        return fn(*self.lambda_args)

    def _resolve_with_args(self, *lambda_args: Any) -> ClauseElement:
        assert isinstance(self._rec, AnalyzedFunction)
        tracker_fn = self._rec.tracker_instrumented_fn
        expr = tracker_fn(*lambda_args)
        expr = coercions.expect(self.role, expr)
        expr = self._setup_binds_for_tracked_expr(expr)
        for deferred_copy_internals in self._transforms:
            expr = deferred_copy_internals(expr)
        return expr

    def _copy_internals(self, clone=_clone, deferred_copy_internals=None, **kw):
        super()._copy_internals(clone=clone, deferred_copy_internals=deferred_copy_internals, opts=kw)
        if deferred_copy_internals:
            self._transforms += (deferred_copy_internals,)