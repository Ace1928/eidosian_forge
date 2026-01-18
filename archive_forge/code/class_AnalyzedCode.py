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
class AnalyzedCode:
    __slots__ = ('track_closure_variables', 'track_bound_values', 'bindparam_trackers', 'closure_trackers', 'build_py_wrappers')
    _fns: weakref.WeakKeyDictionary[CodeType, AnalyzedCode] = weakref.WeakKeyDictionary()
    _generation_mutex = threading.RLock()

    @classmethod
    def get(cls, fn, lambda_element, lambda_kw, **kw):
        try:
            return cls._fns[fn.__code__]
        except KeyError:
            pass
        with cls._generation_mutex:
            if fn.__code__ in cls._fns:
                return cls._fns[fn.__code__]
            analyzed: AnalyzedCode
            cls._fns[fn.__code__] = analyzed = AnalyzedCode(fn, lambda_element, lambda_kw, **kw)
            return analyzed

    def __init__(self, fn, lambda_element, opts):
        if inspect.ismethod(fn):
            raise exc.ArgumentError('Method %s may not be passed as a SQL expression' % fn)
        closure = fn.__closure__
        self.track_bound_values = opts.track_bound_values and opts.global_track_bound_values
        enable_tracking = opts.enable_tracking
        track_on = opts.track_on
        track_closure_variables = opts.track_closure_variables
        self.track_closure_variables = track_closure_variables and (not track_on)
        self.bindparam_trackers = []
        self.closure_trackers = []
        self.build_py_wrappers = []
        if enable_tracking:
            if track_on:
                self._init_track_on(track_on)
            self._init_globals(fn)
            if closure:
                self._init_closure(fn)
        self._setup_additional_closure_trackers(fn, lambda_element, opts)

    def _init_track_on(self, track_on):
        self.closure_trackers.extend((self._cache_key_getter_track_on(idx, elem) for idx, elem in enumerate(track_on)))

    def _init_globals(self, fn):
        build_py_wrappers = self.build_py_wrappers
        bindparam_trackers = self.bindparam_trackers
        track_bound_values = self.track_bound_values
        for name in fn.__code__.co_names:
            if name not in fn.__globals__:
                continue
            _bound_value = self._roll_down_to_literal(fn.__globals__[name])
            if coercions._deep_is_literal(_bound_value):
                build_py_wrappers.append((name, None))
                if track_bound_values:
                    bindparam_trackers.append(self._bound_parameter_getter_func_globals(name))

    def _init_closure(self, fn):
        build_py_wrappers = self.build_py_wrappers
        closure = fn.__closure__
        track_bound_values = self.track_bound_values
        track_closure_variables = self.track_closure_variables
        bindparam_trackers = self.bindparam_trackers
        closure_trackers = self.closure_trackers
        for closure_index, (fv, cell) in enumerate(zip(fn.__code__.co_freevars, closure)):
            _bound_value = self._roll_down_to_literal(cell.cell_contents)
            if coercions._deep_is_literal(_bound_value):
                build_py_wrappers.append((fv, closure_index))
                if track_bound_values:
                    bindparam_trackers.append(self._bound_parameter_getter_func_closure(fv, closure_index))
            elif track_closure_variables:
                closure_trackers.append(self._cache_key_getter_closure_variable(fn, fv, closure_index, cell.cell_contents))

    def _setup_additional_closure_trackers(self, fn, lambda_element, opts):
        analyzed_function = AnalyzedFunction(self, lambda_element, None, fn)
        closure_trackers = self.closure_trackers
        for pywrapper in analyzed_function.closure_pywrappers:
            if not pywrapper._sa__has_param:
                closure_trackers.append(self._cache_key_getter_tracked_literal(fn, pywrapper))

    @classmethod
    def _roll_down_to_literal(cls, element):
        is_clause_element = hasattr(element, '__clause_element__')
        if is_clause_element:
            while not isinstance(element, (elements.ClauseElement, schema.SchemaItem, type)):
                try:
                    element = element.__clause_element__()
                except AttributeError:
                    break
        if not is_clause_element:
            insp = inspection.inspect(element, raiseerr=False)
            if insp is not None:
                try:
                    return insp.__clause_element__()
                except AttributeError:
                    return insp
            return element
        else:
            return element

    def _bound_parameter_getter_func_globals(self, name):
        """Return a getter that will extend a list of bound parameters
        with new entries from the ``__globals__`` collection of a particular
        lambda.

        """

        def extract_parameter_value(current_fn, tracker_instrumented_fn, result):
            wrapper = tracker_instrumented_fn.__globals__[name]
            object.__getattribute__(wrapper, '_extract_bound_parameters')(current_fn.__globals__[name], result)
        return extract_parameter_value

    def _bound_parameter_getter_func_closure(self, name, closure_index):
        """Return a getter that will extend a list of bound parameters
        with new entries from the ``__closure__`` collection of a particular
        lambda.

        """

        def extract_parameter_value(current_fn, tracker_instrumented_fn, result):
            wrapper = tracker_instrumented_fn.__closure__[closure_index].cell_contents
            object.__getattribute__(wrapper, '_extract_bound_parameters')(current_fn.__closure__[closure_index].cell_contents, result)
        return extract_parameter_value

    def _cache_key_getter_track_on(self, idx, elem):
        """Return a getter that will extend a cache key with new entries
        from the "track_on" parameter passed to a :class:`.LambdaElement`.

        """
        if isinstance(elem, tuple):

            def get(closure, opts, anon_map, bindparams):
                return tuple((tup_elem._gen_cache_key(anon_map, bindparams) for tup_elem in opts.track_on[idx]))
        elif isinstance(elem, _cache_key.HasCacheKey):

            def get(closure, opts, anon_map, bindparams):
                return opts.track_on[idx]._gen_cache_key(anon_map, bindparams)
        else:

            def get(closure, opts, anon_map, bindparams):
                return opts.track_on[idx]
        return get

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

    def _raise_for_uncacheable_closure_variable(self, variable_name, fn, from_=None):
        raise exc.InvalidRequestError("Closure variable named '%s' inside of lambda callable %s does not refer to a cacheable SQL element, and also does not appear to be serving as a SQL literal bound value based on the default SQL expression returned by the function.   This variable needs to remain outside the scope of a SQL-generating lambda so that a proper cache key may be generated from the lambda's state.  Evaluate this variable outside of the lambda, set track_on=[<elements>] to explicitly select closure elements to track, or set track_closure_variables=False to exclude closure variables from being part of the cache key." % (variable_name, fn.__code__)) from from_

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