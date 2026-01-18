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
def _convert_glyphs(value: dict[str, Glyph] | Sequence[Glyph]) -> dict[str, Glyph]:
    result: dict[str, Glyph] = {}
    if isinstance(value, dict):
        glyph_ids = set()
        for name, glyph in value.items():
            if not isinstance(glyph, Glyph):
                raise TypeError(f'Expected Glyph, found {type(glyph).__name__}')
            if glyph is not _GLYPH_NOT_LOADED:
                glyph_id = id(glyph)
                if glyph_id in glyph_ids:
                    raise KeyError(f"{glyph!r} can't be added twice")
                glyph_ids.add(glyph_id)
                if glyph.name is None:
                    glyph._name = name
                elif glyph.name != name:
                    raise ValueError(f"glyph has incorrect name: expected '{name}', found '{glyph.name}'")
            result[name] = glyph
    else:
        for glyph in value:
            if not isinstance(glyph, Glyph):
                raise TypeError(f'Expected Glyph, found {type(glyph).__name__}')
            if glyph.name is None:
                raise ValueError(f"{glyph!r} has no name; can't add it to Layer")
            if glyph.name in result:
                raise KeyError(f"glyph named '{glyph.name}' already exists")
            result[glyph.name] = glyph
    return result