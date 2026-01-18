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
def _retrieve_tracker_rec(self, fn, apply_propagate_attrs, opts):
    lambda_cache = opts.lambda_cache
    if lambda_cache is None:
        lambda_cache = _closure_per_cache_key
    tracker_key = self.tracker_key
    fn = self.fn
    closure = fn.__closure__
    tracker = AnalyzedCode.get(fn, self, opts)
    bindparams: List[BindParameter[Any]]
    self._resolved_bindparams = bindparams = []
    if self.parent_lambda is not None:
        parent_closure_cache_key = self.parent_lambda.closure_cache_key
    else:
        parent_closure_cache_key = ()
    cache_key: Union[Tuple[Any, ...], Literal[CacheConst.NO_CACHE]]
    if parent_closure_cache_key is not _cache_key.NO_CACHE:
        anon_map = visitors.anon_map()
        cache_key = tuple([getter(closure, opts, anon_map, bindparams) for getter in tracker.closure_trackers])
        if _cache_key.NO_CACHE not in anon_map:
            cache_key = parent_closure_cache_key + cache_key
            self.closure_cache_key = cache_key
            try:
                rec = lambda_cache[tracker_key + cache_key]
            except KeyError:
                rec = None
        else:
            cache_key = _cache_key.NO_CACHE
            rec = None
    else:
        cache_key = _cache_key.NO_CACHE
        rec = None
    self.closure_cache_key = cache_key
    if rec is None:
        if cache_key is not _cache_key.NO_CACHE:
            with AnalyzedCode._generation_mutex:
                key = tracker_key + cache_key
                if key not in lambda_cache:
                    rec = AnalyzedFunction(tracker, self, apply_propagate_attrs, fn)
                    rec.closure_bindparams = bindparams
                    lambda_cache[key] = rec
                else:
                    rec = lambda_cache[key]
        else:
            rec = NonAnalyzedFunction(self._invoke_user_fn(fn))
    else:
        bindparams[:] = [orig_bind._with_value(new_bind.value, maintain_key=True) for orig_bind, new_bind in zip(rec.closure_bindparams, bindparams)]
    self._rec = rec
    if cache_key is not _cache_key.NO_CACHE:
        if self.parent_lambda is not None:
            bindparams[:0] = self.parent_lambda._resolved_bindparams
        lambda_element: Optional[LambdaElement] = self
        while lambda_element is not None:
            rec = lambda_element._rec
            if rec.bindparam_trackers:
                tracker_instrumented_fn = rec.tracker_instrumented_fn
                for tracker in rec.bindparam_trackers:
                    tracker(lambda_element.fn, tracker_instrumented_fn, bindparams)
            lambda_element = lambda_element.parent_lambda
    return rec