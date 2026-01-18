from __future__ import annotations
from typing import (
from attrs import define, field
from fontTools.ufoLib import UFOReader, UFOWriter
from ufoLib2.constants import DEFAULT_LAYER_NAME
from ufoLib2.errors import Error
from ufoLib2.objects.layer import Layer
from ufoLib2.objects.misc import (
from ufoLib2.serde import serde
from ufoLib2.typing import T
def _must_have_at_least_one_item(self: Any, attribute: Any, value: Sized) -> None:
    if not len(value):
        raise ValueError('value must have at least one item.')