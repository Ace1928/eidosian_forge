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
def _convert_GaspBehavior(seq: Sequence[GaspBehavior | int]) -> list[GaspBehavior]:
    return [v if isinstance(v, GaspBehavior) else GaspBehavior(v) for v in seq]