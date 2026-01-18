from __future__ import annotations
import argparse
import dataclasses
import enum
import functools
import itertools
import json
import shlex
from typing import (
import rich.markup
import shtab
from . import _fields, _instantiators, _resolver, _strings
from ._typing import TypeForm
from .conf import _markers
def _rule_convert_defaults_to_strings(arg: ArgumentDefinition, lowered: LoweredArgumentDefinition) -> LoweredArgumentDefinition:
    """Sets all default values to strings, as required as input to our instantiator
    functions. Special-cased for enums."""

    def as_str(x: Any) -> Tuple[str, ...]:
        if isinstance(x, str):
            return (x,)
        elif isinstance(x, enum.Enum):
            return (x.name,)
        elif isinstance(x, Mapping):
            return tuple(itertools.chain(*map(as_str, itertools.chain(*x.items()))))
        elif isinstance(x, Sequence):
            return tuple(itertools.chain(*map(as_str, x)))
        else:
            return (str(x),)
    if lowered.default is None or lowered.default in _fields.MISSING_SINGLETONS or lowered.action is not None:
        return lowered
    else:
        return dataclasses.replace(lowered, default=as_str(lowered.default))