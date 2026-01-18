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
def _raise_for_uncacheable_closure_variable(self, variable_name, fn, from_=None):
    raise exc.InvalidRequestError("Closure variable named '%s' inside of lambda callable %s does not refer to a cacheable SQL element, and also does not appear to be serving as a SQL literal bound value based on the default SQL expression returned by the function.   This variable needs to remain outside the scope of a SQL-generating lambda so that a proper cache key may be generated from the lambda's state.  Evaluate this variable outside of the lambda, set track_on=[<elements>] to explicitly select closure elements to track, or set track_closure_variables=False to exclude closure variables from being part of the cache key." % (variable_name, fn.__code__)) from from_