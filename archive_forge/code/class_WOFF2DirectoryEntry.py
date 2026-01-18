from io import BytesIO
import sys
import array
import struct
from collections import OrderedDict
from fontTools.misc import sstruct
from fontTools.misc.arrayTools import calcIntBounds
from fontTools.misc.textTools import Tag, bytechr, byteord, bytesjoin, pad
from fontTools.ttLib import (
from fontTools.ttLib.sfnt import (
from fontTools.ttLib.tables import ttProgram, _g_l_y_f
import logging
class WOFF2DirectoryEntry(DirectoryEntry):

    def fromFile(self, file):
        pos = file.tell()
        data = file.read(woff2DirectoryEntryMaxSize)
        left = self.fromString(data)
        consumed = len(data) - len(left)
        file.seek(pos + consumed)

    def fromString(self, data):
        if len(data) < 1:
            raise TTLibError("can't read table 'flags': not enough data")
        dummy, data = sstruct.unpack2(woff2FlagsFormat, data, self)
        if self.flags & 63 == 63:
            if len(data) < woff2UnknownTagSize:
                raise TTLibError("can't read table 'tag': not enough data")
            dummy, data = sstruct.unpack2(woff2UnknownTagFormat, data, self)
        else:
            self.tag = woff2KnownTags[self.flags & 63]
        self.tag = Tag(self.tag)
        self.origLength, data = unpackBase128(data)
        self.length = self.origLength
        if self.transformed:
            self.length, data = unpackBase128(data)
            if self.tag == 'loca' and self.length != 0:
                raise TTLibError("the transformLength of the 'loca' table must be 0")
        return data

    def toString(self):
        data = bytechr(self.flags)
        if self.flags & 63 == 63:
            data += struct.pack('>4s', self.tag.tobytes())
        data += packBase128(self.origLength)
        if self.transformed:
            data += packBase128(self.length)
        return data

    @property
    def transformVersion(self):
        """Return bits 6-7 of table entry's flags, which indicate the preprocessing
        transformation version number (between 0 and 3).
        """
        return self.flags >> 6

    @transformVersion.setter
    def transformVersion(self, value):
        assert 0 <= value <= 3
        self.flags |= value << 6

    @property
    def transformed(self):
        """Return True if the table has any transformation, else return False."""
        if self.tag in {'glyf', 'loca'}:
            return self.transformVersion != 3
        else:
            return self.transformVersion != 0

    @transformed.setter
    def transformed(self, booleanValue):
        if self.tag in {'glyf', 'loca'}:
            self.transformVersion = 3 if not booleanValue else 0
        else:
            self.transformVersion = int(booleanValue)