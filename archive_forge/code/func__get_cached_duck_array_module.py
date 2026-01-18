from __future__ import annotations
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from packaging.version import Version
from xarray.core.utils import is_scalar
from xarray.namedarray.utils import is_duck_array, is_duck_dask_array
def _get_cached_duck_array_module(mod: ModType) -> DuckArrayModule:
    if mod not in _cached_duck_array_modules:
        duckmod = DuckArrayModule(mod)
        _cached_duck_array_modules[mod] = duckmod
        return duckmod
    else:
        return _cached_duck_array_modules[mod]