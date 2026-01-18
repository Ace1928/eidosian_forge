from __future__ import annotations
from typing import TYPE_CHECKING
from typing import NamedTuple
from types import ModuleType
import inspect
from ._helpers import _check_device, _is_numpy_array, array_namespace
def _unique_kwargs(xp):
    s = inspect.signature(xp.unique)
    if 'equal_nan' in s.parameters:
        return {'equal_nan': False}
    return {}