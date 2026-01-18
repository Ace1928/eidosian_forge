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
def _positive(instance: Any, attribute: Any, value: int) -> None:
    if value < 0:
        raise ValueError("'{name}' must be at least 0 (got {value!r})".format(name=attribute.name, value=value))