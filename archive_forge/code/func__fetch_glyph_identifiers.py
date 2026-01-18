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
def _fetch_glyph_identifiers(glyph: Glyph) -> set[str]:
    """Returns all identifiers in use in a glyph."""
    identifiers = set()
    for anchor in glyph.anchors:
        if anchor.identifier is not None:
            identifiers.add(anchor.identifier)
    for guideline in glyph.guidelines:
        if guideline.identifier is not None:
            identifiers.add(guideline.identifier)
    for contour in glyph.contours:
        if contour.identifier is not None:
            identifiers.add(contour.identifier)
        for point in contour:
            if point.identifier is not None:
                identifiers.add(point.identifier)
    for component in glyph.components:
        if component.identifier is not None:
            identifiers.add(component.identifier)
    return identifiers