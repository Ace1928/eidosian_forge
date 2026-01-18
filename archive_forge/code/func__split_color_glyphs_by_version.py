import collections
import copy
import enum
from functools import partial
from math import ceil, log
from typing import (
from fontTools.misc.arrayTools import intRect
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.treeTools import build_n_ary_tree
from fontTools.ttLib.tables import C_O_L_R_
from fontTools.ttLib.tables import C_P_A_L_
from fontTools.ttLib.tables import _n_a_m_e
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otTables import ExtendMode, CompositeMode
from .errors import ColorLibError
from .geometry import round_start_circle_stable_containment
from .table_builder import BuildCallback, TableBuilder
def _split_color_glyphs_by_version(colorGlyphs: _ColorGlyphsDict) -> Tuple[_ColorGlyphsV0Dict, _ColorGlyphsDict]:
    colorGlyphsV0 = {}
    colorGlyphsV1 = {}
    for baseGlyph, layers in colorGlyphs.items():
        if all((_is_colrv0_layer(l) for l in layers)):
            colorGlyphsV0[baseGlyph] = layers
        else:
            colorGlyphsV1[baseGlyph] = layers
    assert set(colorGlyphs) == set(colorGlyphsV0) | set(colorGlyphsV1)
    return (colorGlyphsV0, colorGlyphsV1)