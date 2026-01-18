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
def _instantiator_from_container_type(typ: TypeForm[Any], type_from_typevar: Dict[TypeVar, TypeForm[Any]], markers: FrozenSet[_markers.Marker]) -> Optional[Tuple[Instantiator, InstantiatorMetadata]]:
    """Attempt to create an instantiator from a container type. Returns `None` is no
    container type is found."""
    if typ in (dict, Dict):
        typ = Dict[str, str]
    elif typ in (tuple, Tuple):
        typ = Tuple[str, ...]
    elif typ in (list, List, collections.abc.Sequence, Sequence):
        typ = List[str]
    elif typ in (set, Set):
        typ = Set[str]
    type_origin = get_origin(typ)
    if type_origin is None:
        return None
    if type_origin in (Annotated, Final):
        contained_type = get_args(typ)[0]
        return instantiator_from_type(contained_type, type_from_typevar, markers)
    for make, matched_origins in {_instantiator_from_sequence: (collections.abc.Sequence, frozenset, list, set, deque), _instantiator_from_tuple: (tuple,), _instantiator_from_dict: (dict, collections.abc.Mapping), _instantiator_from_union: (Union,), _instantiator_from_literal: (Literal, LiteralAlternate)}.items():
        if type_origin in matched_origins:
            return make(typ, type_from_typevar, markers)
    raise UnsupportedTypeAnnotationError(f'Unsupported type {typ} with origin {type_origin}')