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
def _rule_static_cast_choices_to_patched_list(arg: ArgumentDefinition, lowered: LoweredArgumentDefinition) -> LoweredArgumentDefinition:
    return dataclasses.replace(lowered, choices=_PatchedList(lowered.choices) if lowered.choices is not None else None)