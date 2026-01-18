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
def _rule_generate_helptext(arg: ArgumentDefinition, lowered: LoweredArgumentDefinition) -> LoweredArgumentDefinition:
    """Generate helptext from docstring, argument name, default values."""
    if _markers.Suppress in arg.field.markers or (_markers.SuppressFixed in arg.field.markers and lowered.is_fixed()):
        return dataclasses.replace(lowered, help=argparse.SUPPRESS)
    help_parts = []
    primary_help = arg.field.helptext
    if primary_help is None and _markers.Positional in arg.field.markers:
        primary_help = _strings.make_field_name([arg.extern_prefix, arg.field.intern_name])
    if primary_help is not None and primary_help != '':
        help_parts.append(_rich_tag_if_enabled(primary_help, 'helptext'))
    if not lowered.required:
        default = lowered.default
        if lowered.is_fixed() or lowered.action == 'append':
            assert default in _fields.MISSING_SINGLETONS or default is None
            default = arg.field.default
        if arg.field.argconf.constructor_factory is not None:
            default_label = str(default) if arg.field.type_or_callable is not json.loads else json.dumps(arg.field.default)
        elif hasattr(default, '__iter__'):
            assert default is not None
            default_label = ' '.join(map(shlex.quote, map(str, default)))
        else:
            default_label = str(default)
        if lowered.instantiator is None:
            default_text = f'(fixed to: {default_label})'
        elif lowered.action == 'append' and (default in _fields.MISSING_SINGLETONS or len(cast(tuple, default)) == 0):
            default_text = '(repeatable)'
        elif lowered.action == 'append' and len(cast(tuple, default)) > 0:
            assert default is not None
            default_text = f'(repeatable, appends to: {default_label})'
        elif arg.field.default is _fields.EXCLUDE_FROM_CALL:
            default_text = '(unset by default)'
        elif _markers._OPTIONAL_GROUP in arg.field.markers and default in _fields.MISSING_SINGLETONS:
            default_text = '(optional)'
        elif _markers._OPTIONAL_GROUP in arg.field.markers:
            default_text = f'(default if used: {default_label})'
        else:
            default_text = f'(default: {default_label})'
        help_parts.append(_rich_tag_if_enabled(default_text, 'helptext_default'))
    else:
        help_parts.append(_rich_tag_if_enabled('(required)', 'helptext_required'))
    return dataclasses.replace(lowered, help=' '.join(help_parts).replace('%', '%%'))