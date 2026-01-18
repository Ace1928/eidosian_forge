from __future__ import annotations
from enum import IntEnum
from functools import partial
from typing import Any, Callable, List, Mapping, Optional, Sequence, TypeVar
import attrs
from attrs import define, field
from fontTools.ufoLib import UFOReader
from ufoLib2.objects.guideline import Guideline
from ufoLib2.objects.misc import AttrDictMixin
from ufoLib2.serde import serde
from .woff import (
class WidthClass(IntEnum):
    ULTRA_CONDENSED = 1
    EXTRA_CONDESED = 2
    CONDENSED = 3
    SEMI_CONDENSED = 4
    NORMAL = 5
    MEDIUM = 5
    SEMI_EXPANDED = 6
    EXPANDED = 7
    EXTRA_EXPANDED = 8
    ULTRA_EXPANDED = 9