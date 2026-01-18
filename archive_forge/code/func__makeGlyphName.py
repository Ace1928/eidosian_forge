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
@staticmethod
def _makeGlyphName(codepoint):
    from fontTools import agl
    if codepoint in agl.UV2AGL:
        return agl.UV2AGL[codepoint]
    elif codepoint <= 65535:
        return 'uni%04X' % codepoint
    else:
        return 'u%X' % codepoint