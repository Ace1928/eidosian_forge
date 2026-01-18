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
def _cache_key_getter_closure_variable(self, fn, variable_name, idx, cell_contents, use_clause_element=False, use_inspect=False):
    """Return a getter that will extend a cache key with new entries
        from the ``__closure__`` collection of a particular lambda.

        """
    if isinstance(cell_contents, _cache_key.HasCacheKey):

        def get(closure, opts, anon_map, bindparams):
            obj = closure[idx].cell_contents
            if use_inspect:
                obj = inspection.inspect(obj)
            elif use_clause_element:
                while hasattr(obj, '__clause_element__'):
                    if not getattr(obj, 'is_clause_element', False):
                        obj = obj.__clause_element__()
            return obj._gen_cache_key(anon_map, bindparams)
    elif isinstance(cell_contents, types.FunctionType):

        def get(closure, opts, anon_map, bindparams):
            return closure[idx].cell_contents.__code__
    elif isinstance(cell_contents, collections_abc.Sequence):

        def get(closure, opts, anon_map, bindparams):
            contents = closure[idx].cell_contents
            try:
                return tuple((elem._gen_cache_key(anon_map, bindparams) for elem in contents))
            except AttributeError as ae:
                self._raise_for_uncacheable_closure_variable(variable_name, fn, from_=ae)
    else:
        element = cell_contents
        is_clause_element = False
        while hasattr(element, '__clause_element__'):
            is_clause_element = True
            if not getattr(element, 'is_clause_element', False):
                element = element.__clause_element__()
            else:
                break
        if not is_clause_element:
            insp = inspection.inspect(element, raiseerr=False)
            if insp is not None:
                return self._cache_key_getter_closure_variable(fn, variable_name, idx, insp, use_inspect=True)
        else:
            return self._cache_key_getter_closure_variable(fn, variable_name, idx, element, use_clause_element=True)
        self._raise_for_uncacheable_closure_variable(variable_name, fn)
    return get