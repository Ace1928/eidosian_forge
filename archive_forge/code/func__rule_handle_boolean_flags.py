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
def _rule_handle_boolean_flags(arg: ArgumentDefinition, lowered: LoweredArgumentDefinition) -> LoweredArgumentDefinition:
    if _resolver.apply_type_from_typevar(arg.field.type_or_callable, arg.type_from_typevar) is not bool:
        return lowered
    if arg.field.default in _fields.MISSING_SINGLETONS or arg.field.is_positional() or _markers.FlagConversionOff in arg.field.markers or (_markers.Fixed in arg.field.markers):
        return lowered
    elif arg.field.default in (True, False):
        return dataclasses.replace(lowered, action=BooleanOptionalAction, instantiator=lambda x: x)
    assert False, f'Expected a boolean as a default for {arg.field.intern_name}, but got {lowered.default}.'