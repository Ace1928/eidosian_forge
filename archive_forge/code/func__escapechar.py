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
def _escapechar(c):
    """Helper function for tagToIdentifier()"""
    import re
    if re.match('[a-z0-9]', c):
        return '_' + c
    elif re.match('[A-Z]', c):
        return c + '_'
    else:
        return hex(byteord(c))[2:]