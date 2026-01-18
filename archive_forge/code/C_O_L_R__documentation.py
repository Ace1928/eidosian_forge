from fontTools.misc.textTools import safeEval
from . import DefaultTable
This table is structured so that you can treat it like a dictionary keyed by glyph name.

    ``ttFont['COLR'][<glyphName>]`` will return the color layers for any glyph.

    ``ttFont['COLR'][<glyphName>] = <value>`` will set the color layers for any glyph.
    