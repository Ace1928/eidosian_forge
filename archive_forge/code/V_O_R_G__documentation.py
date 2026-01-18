from fontTools.misc.textTools import bytesjoin, safeEval
from . import DefaultTable
import struct
This table is structured so that you can treat it like a dictionary keyed by glyph name.

    ``ttFont['VORG'][<glyphName>]`` will return the vertical origin for any glyph.

    ``ttFont['VORG'][<glyphName>] = <value>`` will set the vertical origin for any glyph.
    