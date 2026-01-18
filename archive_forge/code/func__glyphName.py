import logging
import re
from io import StringIO
from fontTools.feaLib import ast
from fontTools.ttLib import TTFont, TTLibError
from fontTools.voltLib import ast as VAst
from fontTools.voltLib.parser import Parser as VoltParser
def _glyphName(self, glyph):
    try:
        name = glyph.glyph
    except AttributeError:
        name = glyph
    return ast.GlyphName(self._glyph_map.get(name, name))