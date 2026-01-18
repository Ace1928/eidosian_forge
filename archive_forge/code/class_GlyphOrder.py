from fontTools.config import Config
from fontTools.misc import xmlWriter
from fontTools.misc.configTools import AbstractConfig
from fontTools.misc.textTools import Tag, byteord, tostr
from fontTools.misc.loggingTools import deprecateArgument
from fontTools.ttLib import TTLibError
from fontTools.ttLib.ttGlyphSet import _TTGlyph, _TTGlyphSetCFF, _TTGlyphSetGlyf
from fontTools.ttLib.sfnt import SFNTReader, SFNTWriter
from io import BytesIO, StringIO, UnsupportedOperation
import os
import logging
import traceback
class GlyphOrder(object):
    """A pseudo table. The glyph order isn't in the font as a separate
    table, but it's nice to present it as such in the TTX format.
    """

    def __init__(self, tag=None):
        pass

    def toXML(self, writer, ttFont):
        glyphOrder = ttFont.getGlyphOrder()
        writer.comment("The 'id' attribute is only for humans; it is ignored when parsed.")
        writer.newline()
        for i in range(len(glyphOrder)):
            glyphName = glyphOrder[i]
            writer.simpletag('GlyphID', id=i, name=glyphName)
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if not hasattr(self, 'glyphOrder'):
            self.glyphOrder = []
        if name == 'GlyphID':
            self.glyphOrder.append(attrs['name'])
        ttFont.setGlyphOrder(self.glyphOrder)