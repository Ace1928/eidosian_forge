from __future__ import annotations
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from packaging.version import Version
from xarray.core.utils import is_scalar
from xarray.namedarray.utils import is_duck_array, is_duck_dask_array
class DuckArrayModule:
    """
    Solely for internal isinstance and version checks.

    Motivated by having to only import pint when required (as pint currently imports xarray)
    https://github.com/pydata/xarray/pull/5561#discussion_r664815718
    """
    module: ModuleType | None
    version: Version
    type: DuckArrayTypes
    available: bool

    def __init__(self, mod: ModType) -> None:
        duck_array_module: ModuleType | None
        duck_array_version: Version
        duck_array_type: DuckArrayTypes
        try:
            duck_array_module = import_module(mod)
            duck_array_version = Version(duck_array_module.__version__)
            if mod == 'dask':
                duck_array_type = (import_module('dask.array').Array,)
            elif mod == 'pint':
                duck_array_type = (duck_array_module.Quantity,)
            elif mod == 'cupy':
                duck_array_type = (duck_array_module.ndarray,)
            elif mod == 'sparse':
                duck_array_type = (duck_array_module.SparseArray,)
            elif mod == 'cubed':
                duck_array_type = (duck_array_module.Array,)
            elif mod == 'numbagg':
                duck_array_type = ()
            else:
                raise NotImplementedError
        except (ImportError, AttributeError):
            duck_array_module = None
            duck_array_version = Version('0.0.0')
            duck_array_type = ()
        self.module = duck_array_module
        self.version = duck_array_version
        self.type = duck_array_type
        self.available = duck_array_module is not None