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
class MockFont(object):
    """A font-like object that automatically adds any looked up glyphname
    to its glyphOrder."""

    def __init__(self):
        self._glyphOrder = ['.notdef']

        class AllocatingDict(dict):

            def __missing__(reverseDict, key):
                self._glyphOrder.append(key)
                gid = len(reverseDict)
                reverseDict[key] = gid
                return gid
        self._reverseGlyphOrder = AllocatingDict({'.notdef': 0})
        self.lazy = False

    def getGlyphID(self, glyph):
        gid = self._reverseGlyphOrder[glyph]
        return gid

    def getReverseGlyphMap(self):
        return self._reverseGlyphOrder

    def getGlyphName(self, gid):
        return self._glyphOrder[gid]

    def getGlyphOrder(self):
        return self._glyphOrder