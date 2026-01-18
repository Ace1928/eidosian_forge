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
def _instantiator_from_literal(typ: TypeForm, type_from_typevar: Dict[TypeVar, TypeForm[Any]], markers: FrozenSet[_markers.Marker]) -> Tuple[_StandardInstantiator, InstantiatorMetadata]:
    choices = get_args(typ)
    str_choices = tuple((x.name if isinstance(x, enum.Enum) else str(x) for x in choices))
    return (lambda strings: choices[str_choices.index(strings[0])], InstantiatorMetadata(nargs=1, metavar='{' + ','.join(str_choices) + '}', choices=set(str_choices), action=None))