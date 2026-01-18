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
class _PatchedList(list):
    """Custom list type, for avoiding "default not in choices" errors when the default
    is set to MISSING_NONPROP.

    This solves a choices error raised by argparse in a very specific edge case:
    literals in containers as positional arguments."""

    def __init__(self, li):
        super(_PatchedList, self).__init__(li)

    def __contains__(self, x: Any) -> bool:
        return list.__contains__(self, x) or x is _fields.MISSING_NONPROP