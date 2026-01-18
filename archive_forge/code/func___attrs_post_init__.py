from __future__ import annotations
from typing import (
from attrs import define, field
from fontTools.ufoLib.glifLib import GlyphSet
from ufoLib2.constants import DEFAULT_LAYER_NAME
from ufoLib2.objects.glyph import Glyph
from ufoLib2.objects.lib import (
from ufoLib2.objects.misc import (
from ufoLib2.serde import serde
from ufoLib2.typing import T
def __attrs_post_init__(self) -> None:
    if self._name == DEFAULT_LAYER_NAME and (not self._default):
        self._default = True