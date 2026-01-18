import collections.abc
import dataclasses
import enum
import inspect
import os
import pathlib
from collections import deque
from typing import (
from typing_extensions import Annotated, Final, Literal, get_args, get_origin
from . import _resolver
from . import _strings
from ._typing import TypeForm
from .conf import _markers
def _instantiator_from_type_inner(typ: TypeForm, type_from_typevar: Dict[TypeVar, TypeForm[Any]], allow_sequences: Literal['fixed_length', True, False], markers: FrozenSet[_markers.Marker]) -> Tuple[Instantiator, InstantiatorMetadata]:
    """Thin wrapper over instantiator_from_type, with some extra asserts for catching
    errors."""
    out = instantiator_from_type(typ, type_from_typevar, markers)
    if out[1].nargs == '*':
        assert allow_sequences
        if allow_sequences == 'fixed_length' and (not isinstance(out[1].nargs, int)):
            raise UnsupportedTypeAnnotationError(f'{typ} is a variable-length sequence, which is ambiguous when nested. For nesting variable-length sequences (example: List[List[int]]), `tyro.conf.UseAppendAction` can help resolve ambiguities.')
    return out