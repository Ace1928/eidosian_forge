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
def _rule_set_name_or_flag_and_dest(arg: ArgumentDefinition, lowered: LoweredArgumentDefinition) -> LoweredArgumentDefinition:
    name_or_flag = _strings.make_field_name([arg.extern_prefix, arg.field.extern_name] if arg.field.argconf.prefix_name and _markers.OmitArgPrefixes not in arg.field.markers else [arg.field.extern_name])
    if not arg.field.is_positional():
        name_or_flag = '--' + name_or_flag
    if _markers.OmitArgPrefixes not in arg.field.markers and _markers.Positional not in arg.field.markers and name_or_flag.startswith('--') and (arg.subcommand_prefix != ''):
        strip_prefix = '--' + arg.subcommand_prefix + '.'
        if _markers.OmitSubcommandPrefixes in arg.field.markers:
            assert name_or_flag.startswith(strip_prefix), name_or_flag
            name_or_flag = '--' + name_or_flag[len(strip_prefix):]
    return dataclasses.replace(lowered, name_or_flag=name_or_flag, dest=_strings.make_field_name([arg.intern_prefix, arg.field.intern_name]))