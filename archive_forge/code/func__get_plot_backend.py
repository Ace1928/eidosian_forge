from __future__ import annotations
import importlib
from typing import (
from pandas._config import get_option
from pandas.util._decorators import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.base import PandasObject
def _get_plot_backend(backend: str | None=None):
    """
    Return the plotting backend to use (e.g. `pandas.plotting._matplotlib`).

    The plotting system of pandas uses matplotlib by default, but the idea here
    is that it can also work with other third-party backends. This function
    returns the module which provides a top-level `.plot` method that will
    actually do the plotting. The backend is specified from a string, which
    either comes from the keyword argument `backend`, or, if not specified, from
    the option `pandas.options.plotting.backend`. All the rest of the code in
    this file uses the backend specified there for the plotting.

    The backend is imported lazily, as matplotlib is a soft dependency, and
    pandas can be used without it being installed.

    Notes
    -----
    Modifies `_backends` with imported backend as a side effect.
    """
    backend_str: str = backend or get_option('plotting.backend')
    if backend_str in _backends:
        return _backends[backend_str]
    module = _load_backend(backend_str)
    _backends[backend_str] = module
    return module