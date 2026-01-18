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
class PyWrapper(ColumnOperators):
    """A wrapper object that is injected into the ``__globals__`` and
    ``__closure__`` of a Python function.

    When the function is instrumented with :class:`.PyWrapper` objects, it is
    then invoked just once in order to set up the wrappers.  We look through
    all the :class:`.PyWrapper` objects we made to find the ones that generated
    a :class:`.BindParameter` object, e.g. the expression system interpreted
    something as a literal.   Those positions in the globals/closure are then
    ones that we will look at, each time a new lambda comes in that refers to
    the same ``__code__`` object.   In this way, we keep a single version of
    the SQL expression that this lambda produced, without calling upon the
    Python function that created it more than once, unless its other closure
    variables have changed.   The expression is then transformed to have the
    new bound values embedded into it.

    """

    def __init__(self, fn, name, to_evaluate, closure_index=None, getter=None, track_bound_values=True):
        self.fn = fn
        self._name = name
        self._to_evaluate = to_evaluate
        self._param = None
        self._has_param = False
        self._bind_paths = {}
        self._getter = getter
        self._closure_index = closure_index
        self.track_bound_values = track_bound_values

    def __call__(self, *arg, **kw):
        elem = object.__getattribute__(self, '_to_evaluate')
        value = elem(*arg, **kw)
        if self._sa_track_bound_values and coercions._deep_is_literal(value) and (not isinstance(value, _cache_key.HasCacheKey)):
            name = object.__getattribute__(self, '_name')
            raise exc.InvalidRequestError("Can't invoke Python callable %s() inside of lambda expression argument at %s; lambda SQL constructs should not invoke functions from closure variables to produce literal values since the lambda SQL system normally extracts bound values without actually invoking the lambda or any functions within it.  Call the function outside of the lambda and assign to a local variable that is used in the lambda as a closure variable, or set track_bound_values=False if the return value of this function is used in some other way other than a SQL bound value." % (name, self._sa_fn.__code__))
        else:
            return value

    def operate(self, op, *other, **kwargs):
        elem = object.__getattribute__(self, '_py_wrapper_literal')()
        return op(elem, *other, **kwargs)

    def reverse_operate(self, op, other, **kwargs):
        elem = object.__getattribute__(self, '_py_wrapper_literal')()
        return op(other, elem, **kwargs)

    def _extract_bound_parameters(self, starting_point, result_list):
        param = object.__getattribute__(self, '_param')
        if param is not None:
            param = param._with_value(starting_point, maintain_key=True)
            result_list.append(param)
        for pywrapper in object.__getattribute__(self, '_bind_paths').values():
            getter = object.__getattribute__(pywrapper, '_getter')
            element = getter(starting_point)
            pywrapper._sa__extract_bound_parameters(element, result_list)

    def _py_wrapper_literal(self, expr=None, operator=None, **kw):
        param = object.__getattribute__(self, '_param')
        to_evaluate = object.__getattribute__(self, '_to_evaluate')
        if param is None:
            name = object.__getattribute__(self, '_name')
            self._param = param = elements.BindParameter(name, required=False, unique=True, _compared_to_operator=operator, _compared_to_type=expr.type if expr is not None else None)
            self._has_param = True
        return param._with_value(to_evaluate, maintain_key=True)

    def __bool__(self):
        to_evaluate = object.__getattribute__(self, '_to_evaluate')
        return bool(to_evaluate)

    def __getattribute__(self, key):
        if key.startswith('_sa_'):
            return object.__getattribute__(self, key[4:])
        elif key in ('__clause_element__', 'operate', 'reverse_operate', '_py_wrapper_literal', '__class__', '__dict__'):
            return object.__getattribute__(self, key)
        if key.startswith('__'):
            elem = object.__getattribute__(self, '_to_evaluate')
            return getattr(elem, key)
        else:
            return self._sa__add_getter(key, operator.attrgetter)

    def __iter__(self):
        elem = object.__getattribute__(self, '_to_evaluate')
        return iter(elem)

    def __getitem__(self, key):
        elem = object.__getattribute__(self, '_to_evaluate')
        if not hasattr(elem, '__getitem__'):
            raise AttributeError('__getitem__')
        if isinstance(key, PyWrapper):
            raise exc.InvalidRequestError('Dictionary keys / list indexes inside of a cached lambda must be Python literals only')
        return self._sa__add_getter(key, operator.itemgetter)

    def _add_getter(self, key, getter_fn):
        bind_paths = object.__getattribute__(self, '_bind_paths')
        bind_path_key = (key, getter_fn)
        if bind_path_key in bind_paths:
            return bind_paths[bind_path_key]
        getter = getter_fn(key)
        elem = object.__getattribute__(self, '_to_evaluate')
        value = getter(elem)
        rolled_down_value = AnalyzedCode._roll_down_to_literal(value)
        if coercions._deep_is_literal(rolled_down_value):
            wrapper = PyWrapper(self._sa_fn, key, value, getter=getter)
            bind_paths[bind_path_key] = wrapper
            return wrapper
        else:
            return value