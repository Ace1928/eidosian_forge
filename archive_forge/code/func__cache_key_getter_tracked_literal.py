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
def _cache_key_getter_tracked_literal(self, fn, pytracker):
    """Return a getter that will extend a cache key with new entries
        from the ``__closure__`` collection of a particular lambda.

        this getter differs from _cache_key_getter_closure_variable
        in that these are detected after the function is run, and PyWrapper
        objects have recorded that a particular literal value is in fact
        not being interpreted as a bound parameter.

        """
    elem = pytracker._sa__to_evaluate
    closure_index = pytracker._sa__closure_index
    variable_name = pytracker._sa__name
    return self._cache_key_getter_closure_variable(fn, variable_name, closure_index, elem)