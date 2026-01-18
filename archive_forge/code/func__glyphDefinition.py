import logging
import re
from io import StringIO
from fontTools.feaLib import ast
from fontTools.ttLib import TTFont, TTLibError
from fontTools.voltLib import ast as VAst
from fontTools.voltLib.parser import Parser as VoltParser
def _glyphDefinition(self, glyph):
    try:
        self._glyph_map[glyph.name] = self._glyph_order[glyph.id]
    except TypeError:
        pass
    if glyph.type in ('BASE', 'MARK', 'LIGATURE', 'COMPONENT'):
        if glyph.type not in self._gdef:
            self._gdef[glyph.type] = ast.GlyphClass()
        self._gdef[glyph.type].glyphs.append(self._glyphName(glyph.name))
    if glyph.type == 'MARK':
        self._marks.add(glyph.name)
    elif glyph.type == 'LIGATURE':
        self._ligatures[glyph.name] = glyph.components