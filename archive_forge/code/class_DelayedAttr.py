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
class DelayedAttr(Delayed):
    __slots__ = ('_obj', '_attr')

    def __init__(self, obj, attr):
        key = 'getattr-%s' % tokenize(obj, attr, pure=True)
        super().__init__(key, None)
        self._obj = obj
        self._attr = attr

    def __getattr__(self, attr):
        if attr == 'dtype' and self._attr == 'dtype':
            raise AttributeError('Attribute dtype not found')
        return super().__getattr__(attr)

    @property
    def dask(self):
        layer = {self._key: (getattr, self._obj._key, self._attr)}
        return HighLevelGraph.from_collections(self._key, layer, dependencies=[self._obj])

    def __call__(self, *args, **kwargs):
        return call_function(methodcaller(self._attr), self._attr, (self._obj,) + args, kwargs)