from __future__ import annotations
import dataclasses
import enum
import functools
import typing as T
from . import builder
from .. import mparser
from ..mesonlib import MesonBugException
@dataclasses.dataclass
class IR:
    """Base IR node for Cargo CFG."""