from __future__ import annotations
import dataclasses
from typing import Any, Callable, Dict, Optional, Tuple, Union
from typing_extensions import get_args, get_origin
from . import _fields, _instantiators, _resolver, _typing
from .conf import _confstruct
def _get_type_options(typ: _typing.TypeForm) -> Tuple[_typing.TypeForm, ...]:
    return get_args(typ) if get_origin(typ) is Union else (typ,)