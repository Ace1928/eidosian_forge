import logging
import re
from io import StringIO
from fontTools.feaLib import ast
from fontTools.ttLib import TTFont, TTLibError
from fontTools.voltLib import ast as VAst
from fontTools.voltLib.parser import Parser as VoltParser
def _groupDefinition(self, group):
    name = self._className(group.name)
    glyphs = self._enum(group.enum)
    glyphclass = ast.GlyphClassDefinition(name, glyphs)
    self._glyphclasses[group.name.lower()] = glyphclass