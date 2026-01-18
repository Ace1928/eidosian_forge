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
@openTypeOS2WeightClass.validator
def _validate_weight_class(self, attribute: Any, value: int | None) -> None:
    if value is not None and (value < 1 or value > 1000):
        raise ValueError("'openTypeOS2WeightClass' must be between 1 and 1000")