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
def _rule_positional_special_handling(arg: ArgumentDefinition, lowered: LoweredArgumentDefinition) -> LoweredArgumentDefinition:
    if not arg.field.is_positional():
        return lowered
    metavar = lowered.metavar
    if lowered.required:
        nargs = lowered.nargs
    else:
        if metavar is not None:
            metavar = '[' + metavar + ']'
        if lowered.nargs == 1:
            nargs = '?'
        else:
            nargs = '*'
    return dataclasses.replace(lowered, name_or_flag=_strings.make_field_name([arg.intern_prefix, arg.field.intern_name]), dest=None, required=None, metavar=metavar, nargs=nargs)