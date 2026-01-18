from __future__ import annotations
import functools
import warnings
import numpy as np
import pandas as pd
from dask.dataframe._compat import check_to_pydatetime_deprecation
from dask.utils import derived_from
def _bind_method(cls, pd_cls, attr, min_version=None):

    def func(self, *args, **kwargs):
        return self._function_map(attr, *args, **kwargs)
    func.__name__ = attr
    func.__qualname__ = f'{cls.__name__}.{attr}'
    try:
        func.__wrapped__ = getattr(pd_cls, attr)
    except Exception:
        pass
    setattr(cls, attr, derived_from(pd_cls, version=min_version)(func))