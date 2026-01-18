from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import nearestMultipleShortestRepr, otRound
from fontTools.misc.textTools import bytesjoin, tobytes, tostr, pad, safeEval
from fontTools.ttLib import getSearchRange
from .otBase import (
from .otTables import (
from itertools import zip_longest
from functools import partial
import re
import struct
from typing import Optional
import logging
class CIDGlyphMap(BaseConverter):

    def read(self, reader, font, tableDict):
        numCIDs = reader.readUShort()
        result = {}
        for cid, glyphID in enumerate(reader.readUShortArray(numCIDs)):
            if glyphID != 65535:
                result[cid] = font.getGlyphName(glyphID)
        return result

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        items = {cid: font.getGlyphID(glyph) for cid, glyph in value.items()}
        count = max(items) + 1 if items else 0
        writer.writeUShort(count)
        for cid in range(count):
            writer.writeUShort(items.get(cid, 65535))

    def xmlRead(self, attrs, content, font):
        result = {}
        for eName, eAttrs, _eContent in filter(istuple, content):
            if eName == 'CID':
                result[safeEval(eAttrs['cid'])] = eAttrs['glyph'].strip()
        return result

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.begintag(name, attrs)
        xmlWriter.newline()
        for cid, glyph in sorted(value.items()):
            if glyph is not None and glyph != 65535:
                xmlWriter.simpletag('CID', cid=cid, glyph=glyph)
                xmlWriter.newline()
        xmlWriter.endtag(name)
        xmlWriter.newline()