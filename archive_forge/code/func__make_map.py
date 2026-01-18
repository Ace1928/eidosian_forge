from fontTools.misc.textTools import bytesjoin, safeEval, readHex
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import getSearchRange
from fontTools.unicode import Unicode
from . import DefaultTable
import sys
import struct
import array
import logging
def _make_map(font, chars, gids):
    assert len(chars) == len(gids)
    glyphNames = font.getGlyphNameMany(gids)
    cmap = {}
    for char, gid, name in zip(chars, gids, glyphNames):
        if gid == 0:
            continue
        cmap[char] = name
    return cmap