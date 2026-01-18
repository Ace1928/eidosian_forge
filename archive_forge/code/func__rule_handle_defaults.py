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
def _rule_handle_defaults(arg: ArgumentDefinition, lowered: LoweredArgumentDefinition) -> LoweredArgumentDefinition:
    """Set `required=False` if a default value is set."""
    if arg.field.default in _fields.MISSING_SINGLETONS and _markers._OPTIONAL_GROUP not in arg.field.markers:
        return dataclasses.replace(lowered, default=None, required=True)
    return dataclasses.replace(lowered, default=arg.field.default)