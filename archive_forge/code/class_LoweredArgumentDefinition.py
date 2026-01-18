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
@dataclasses.dataclass(frozen=True)
class LoweredArgumentDefinition:
    """Contains fields meant to be passed directly into argparse."""
    instantiator: Optional[_instantiators.Instantiator] = None

    def is_fixed(self) -> bool:
        """If the instantiator is set to `None`, even after all argument
        transformations, it means that we don't have a valid instantiator for an
        argument. We then mark the argument as 'fixed', with a value always equal to the
        field default."""
        return self.instantiator is None
    name_or_flag: str = ''
    default: Optional[Any] = None
    dest: Optional[str] = None
    required: Optional[bool] = None
    action: Optional[Any] = None
    nargs: Optional[Union[int, str]] = None
    choices: Optional[Union[Set[str], List[str]]] = None
    metavar: Optional[str] = None
    help: Optional[str] = None