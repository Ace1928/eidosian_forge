from collections.abc import Iterable
from io import BytesIO
import os
import re
import shutil
import sys
import tempfile
from unittest import TestCase as _TestCase
from fontTools.config import Config
from fontTools.misc.textTools import tobytes
from fontTools.misc.xmlWriter import XMLWriter
class FakeFont:

    def __init__(self, glyphs):
        self.glyphOrder_ = glyphs
        self.reverseGlyphOrderDict_ = {g: i for i, g in enumerate(glyphs)}
        self.lazy = False
        self.tables = {}
        self.cfg = Config()

    def __getitem__(self, tag):
        return self.tables[tag]

    def __setitem__(self, tag, table):
        self.tables[tag] = table

    def get(self, tag, default=None):
        return self.tables.get(tag, default)

    def getGlyphID(self, name):
        return self.reverseGlyphOrderDict_[name]

    def getGlyphIDMany(self, lst):
        return [self.getGlyphID(gid) for gid in lst]

    def getGlyphName(self, glyphID):
        if glyphID < len(self.glyphOrder_):
            return self.glyphOrder_[glyphID]
        else:
            return 'glyph%.5d' % glyphID

    def getGlyphNameMany(self, lst):
        return [self.getGlyphName(gid) for gid in lst]

    def getGlyphOrder(self):
        return self.glyphOrder_

    def getReverseGlyphMap(self):
        return self.reverseGlyphOrderDict_

    def getGlyphNames(self):
        return sorted(self.getGlyphOrder())