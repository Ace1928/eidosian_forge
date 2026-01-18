import datetime
import functools
import importlib
import re
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import numpy as np
import tree
import xarray as xr
from .. import __version__, utils
from ..rcparams import rcParams
def _make_json_serializable(data: dict) -> dict:
    """Convert `data` with numpy.ndarray-like values to JSON-serializable form."""
    ret = {}
    for key, value in data.items():
        try:
            json.dumps(value)
        except (TypeError, OverflowError):
            pass
        else:
            ret[key] = value
            continue
        if isinstance(value, dict):
            ret[key] = _make_json_serializable(value)
        elif isinstance(value, np.ndarray):
            ret[key] = np.asarray(value).tolist()
        else:
            raise TypeError(f'Value associated with variable `{type(value)}` is not JSON serializable.')
    return ret