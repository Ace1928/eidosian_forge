from __future__ import annotations
import operator
from enum import IntEnum
from ._creation_functions import asarray
from ._dtypes import (
from typing import TYPE_CHECKING, Optional, Tuple, Union, Any, SupportsIndex
import types
import cupy as np
from cupy.cuda import Device as _Device
from cupy.cuda import stream as stream_module
from cupy_backends.cuda.api import runtime
from cupy import array_api
def __array_namespace__(self: Array, /, *, api_version: Optional[str]=None) -> types.ModuleType:
    if api_version is not None and (not api_version.startswith('2021.')):
        raise ValueError(f'Unrecognized array API version: {api_version!r}')
    return array_api