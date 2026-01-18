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
def _convert_gasp_range_records(values: Sequence[GaspRangeRecord | Mapping[str, Any]] | None) -> list[GaspRangeRecord] | None:
    return _convert_optional_list_of_dicts(GaspRangeRecord, values)