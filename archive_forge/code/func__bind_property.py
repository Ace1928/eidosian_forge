from __future__ import annotations
import functools
import warnings
import numpy as np
import pandas as pd
from dask.dataframe._compat import check_to_pydatetime_deprecation
from dask.utils import derived_from
def _bind_property(cls, pd_cls, attr, min_version=None):

    def func(self):
        return self._property_map(attr)
    func.__name__ = attr
    func.__qualname__ = f'{cls.__name__}.{attr}'
    try:
        original_prop = getattr(pd_cls, attr)
        if isinstance(original_prop, property):
            method = original_prop.fget
        elif isinstance(original_prop, functools.cached_property):
            method = original_prop.func
        else:
            method = original_prop
            func.__wrapped__ = method
    except Exception:
        pass
    setattr(cls, attr, property(derived_from(pd_cls, version=min_version)(func)))