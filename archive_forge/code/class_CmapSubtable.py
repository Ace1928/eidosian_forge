from fontTools.misc.textTools import bytesjoin, safeEval, readHex
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import getSearchRange
from fontTools.unicode import Unicode
from . import DefaultTable
import sys
import struct
import array
import logging
class CmapSubtable(object):
    """Base class for all cmap subtable formats.

    Subclasses which handle the individual subtable formats are named
    ``cmap_format_0``, ``cmap_format_2`` etc. Use :py:meth:`getSubtableClass`
    to retrieve the concrete subclass, or :py:meth:`newSubtable` to get a
    new subtable object for a given format.

    The object exposes a ``.cmap`` attribute, which contains a dictionary mapping
    character codepoints to glyph names.
    """

    @staticmethod
    def getSubtableClass(format):
        """Return the subtable class for a format."""
        return cmap_classes.get(format, cmap_format_unknown)

    @staticmethod
    def newSubtable(format):
        """Return a new instance of a subtable for the given format
        ."""
        subtableClass = CmapSubtable.getSubtableClass(format)
        return subtableClass(format)

    def __init__(self, format):
        self.format = format
        self.data = None
        self.ttFont = None
        self.platformID = None
        self.platEncID = None
        self.language = None

    def ensureDecompiled(self, recurse=False):
        if self.data is None:
            return
        self.decompile(None, None)
        self.data = None

    def __getattr__(self, attr):
        if attr[:2] == '__':
            raise AttributeError(attr)
        if self.data is None:
            raise AttributeError(attr)
        self.ensureDecompiled()
        return getattr(self, attr)

    def decompileHeader(self, data, ttFont):
        format, length, language = struct.unpack('>HHH', data[:6])
        assert len(data) == length, 'corrupt cmap table format %d (data length: %d, header length: %d)' % (format, len(data), length)
        self.format = int(format)
        self.length = int(length)
        self.language = int(language)
        self.data = data[6:]
        self.ttFont = ttFont

    def toXML(self, writer, ttFont):
        writer.begintag(self.__class__.__name__, [('platformID', self.platformID), ('platEncID', self.platEncID), ('language', self.language)])
        writer.newline()
        codes = sorted(self.cmap.items())
        self._writeCodes(codes, writer)
        writer.endtag(self.__class__.__name__)
        writer.newline()

    def getEncoding(self, default=None):
        """Returns the Python encoding name for this cmap subtable based on its platformID,
        platEncID, and language.  If encoding for these values is not known, by default
        ``None`` is returned.  That can be overridden by passing a value to the ``default``
        argument.

        Note that if you want to choose a "preferred" cmap subtable, most of the time
        ``self.isUnicode()`` is what you want as that one only returns true for the modern,
        commonly used, Unicode-compatible triplets, not the legacy ones.
        """
        return getEncoding(self.platformID, self.platEncID, self.language, default)

    def isUnicode(self):
        """Returns true if the characters are interpreted as Unicode codepoints."""
        return self.platformID == 0 or (self.platformID == 3 and self.platEncID in [0, 1, 10])

    def isSymbol(self):
        """Returns true if the subtable is for the Symbol encoding (3,0)"""
        return self.platformID == 3 and self.platEncID == 0

    def _writeCodes(self, codes, writer):
        isUnicode = self.isUnicode()
        for code, name in codes:
            writer.simpletag('map', code=hex(code), name=name)
            if isUnicode:
                writer.comment(Unicode[code])
            writer.newline()

    def __lt__(self, other):
        if not isinstance(other, CmapSubtable):
            return NotImplemented
        selfTuple = (getattr(self, 'platformID', None), getattr(self, 'platEncID', None), getattr(self, 'language', None), self.__dict__)
        otherTuple = (getattr(other, 'platformID', None), getattr(other, 'platEncID', None), getattr(other, 'language', None), other.__dict__)
        return selfTuple < otherTuple