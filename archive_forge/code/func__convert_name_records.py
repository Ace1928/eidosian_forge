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
def _convert_name_records(values: Sequence[NameRecord | Mapping[str, Any]] | None) -> list[NameRecord] | None:
    return _convert_optional_list_of_dicts(NameRecord, values)