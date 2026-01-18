from __future__ import annotations
import operator
import types
import uuid
import warnings
from collections.abc import Sequence
from dataclasses import fields, is_dataclass, replace
from functools import partial
from tlz import concat, curry, merge, unique
from dask import config
from dask.base import (
from dask.base import tokenize as _tokenize
from dask.context import globalmethod
from dask.core import flatten, quote
from dask.highlevelgraph import HighLevelGraph
from dask.typing import Graph, NestedKeys
from dask.utils import (
class Delayed(DaskMethodsMixin, OperatorMethodMixin):
    """Represents a value to be computed by dask.

    Equivalent to the output from a single key in a dask graph.
    """
    __slots__ = ('_key', '_dask', '_length', '_layer')

    def __init__(self, key, dsk, length=None, layer=None):
        self._key = key
        self._dask = dsk
        self._length = length
        self._layer = layer or key
        if isinstance(dsk, HighLevelGraph) and self._layer not in dsk.layers:
            raise ValueError(f"Layer {self._layer} not in the HighLevelGraph's layers: {list(dsk.layers)}")

    @property
    def key(self):
        return self._key

    @property
    def dask(self):
        return self._dask

    def __dask_graph__(self) -> Graph:
        return self.dask

    def __dask_keys__(self) -> NestedKeys:
        return [self.key]

    def __dask_layers__(self) -> Sequence[str]:
        return (self._layer,)

    def __dask_tokenize__(self):
        return self.key
    __dask_scheduler__ = staticmethod(DEFAULT_GET)
    __dask_optimize__ = globalmethod(optimize, key='delayed_optimize')

    def __dask_postcompute__(self):
        return (single_key, ())

    def __dask_postpersist__(self):
        return (self._rebuild, ())

    def _rebuild(self, dsk, *, rename=None):
        key = replace_name_in_key(self.key, rename) if rename else self.key
        if isinstance(dsk, HighLevelGraph) and len(dsk.layers) == 1:
            layer = next(iter(dsk.layers))
        else:
            layer = None
        return Delayed(key, dsk, self._length, layer=layer)

    def __repr__(self):
        return f'Delayed({repr(self.key)})'

    def __hash__(self):
        return hash(self.key)

    def __dir__(self):
        return dir(type(self))

    def __getattr__(self, attr):
        if attr.startswith('_'):
            raise AttributeError(f'Attribute {attr} not found')
        if attr == 'visualise':
            warnings.warn('dask.delayed objects have no `visualise` method. Perhaps you meant `visualize`?')
        return DelayedAttr(self, attr)

    def __setattr__(self, attr, val):
        try:
            object.__setattr__(self, attr, val)
        except AttributeError:
            raise TypeError('Delayed objects are immutable')

    def __setitem__(self, index, val):
        raise TypeError('Delayed objects are immutable')

    def __iter__(self):
        if self._length is None:
            raise TypeError('Delayed objects of unspecified length are not iterable')
        for i in range(self._length):
            yield self[i]

    def __len__(self):
        if self._length is None:
            raise TypeError('Delayed objects of unspecified length have no len()')
        return self._length

    def __call__(self, *args, pure=None, dask_key_name=None, **kwargs):
        func = delayed(apply, pure=pure)
        if dask_key_name is not None:
            return func(self, args, kwargs, dask_key_name=dask_key_name)
        return func(self, args, kwargs)

    def __bool__(self):
        raise TypeError('Truth of Delayed objects is not supported')
    __nonzero__ = __bool__

    def __get__(self, instance, cls):
        if instance is None:
            return self
        return types.MethodType(self, instance)

    @classmethod
    def _get_binary_operator(cls, op, inv=False):
        method = delayed(right(op) if inv else op, pure=True)
        return lambda *args, **kwargs: method(*args, **kwargs)
    _get_unary_operator = _get_binary_operator