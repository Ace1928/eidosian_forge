from __future__ import annotations as _annotations
import dataclasses
import sys
import warnings
from copy import copy
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from pydantic_core import PydanticUndefined
from pydantic.errors import PydanticUserError
from . import _typing_extra
from ._config import ConfigWrapper
from ._repr import Representation
from ._typing_extra import get_cls_type_hints_lenient, get_type_hints, is_classvar, is_finalvar
def _is_finalvar_with_default_val(type_: type[Any], val: Any) -> bool:
    from ..fields import FieldInfo
    if not is_finalvar(type_):
        return False
    elif val is PydanticUndefined:
        return False
    elif isinstance(val, FieldInfo) and (val.default is PydanticUndefined and val.default_factory is None):
        return False
    else:
        return True