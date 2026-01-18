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
def _instantiator_from_tuple(typ: TypeForm, type_from_typevar: Dict[TypeVar, TypeForm[Any]], markers: FrozenSet[_markers.Marker]) -> Tuple[Instantiator, InstantiatorMetadata]:
    types = get_args(typ)
    typeset = set(types)
    typeset_no_ellipsis = typeset - {Ellipsis}
    if typeset_no_ellipsis != typeset:
        assert len(typeset_no_ellipsis) == 1
        return _instantiator_from_sequence(typ, type_from_typevar, markers)
    else:
        instantiators: List[_StandardInstantiator] = []
        metas: List[InstantiatorMetadata] = []
        nargs = 0
        for t in types:
            a, b = _instantiator_from_type_inner(t, type_from_typevar, allow_sequences='fixed_length', markers=markers)
            instantiators.append(a)
            metas.append(b)
            assert isinstance(b.nargs, int)
            nargs += b.nargs

        def fixed_length_tuple_instantiator(strings: List[str]) -> Any:
            assert len(strings) == nargs
            out = []
            i = 0
            for make, meta in zip(instantiators, metas):
                assert isinstance(meta.nargs, int)
                meta.check_choices(strings[i:i + meta.nargs])
                out.append(make(strings[i:i + meta.nargs]))
                i += meta.nargs
            return tuple(out)
        return (fixed_length_tuple_instantiator, InstantiatorMetadata(nargs=nargs, metavar=' '.join((m.metavar for m in metas)), choices=None, action=None))