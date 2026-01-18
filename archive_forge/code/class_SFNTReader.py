from io import BytesIO
from types import SimpleNamespace
from fontTools.misc.textTools import Tag
from fontTools.misc import sstruct
from fontTools.ttLib import TTLibError, TTLibFileIsCollectionError
import struct
from collections import OrderedDict
import logging
class SFNTReader(object):

    def __new__(cls, *args, **kwargs):
        """Return an instance of the SFNTReader sub-class which is compatible
        with the input file type.
        """
        if args and cls is SFNTReader:
            infile = args[0]
            infile.seek(0)
            sfntVersion = Tag(infile.read(4))
            infile.seek(0)
            if sfntVersion == 'wOF2':
                from fontTools.ttLib.woff2 import WOFF2Reader
                return object.__new__(WOFF2Reader)
        return object.__new__(cls)

    def __init__(self, file, checkChecksums=0, fontNumber=-1):
        self.file = file
        self.checkChecksums = checkChecksums
        self.flavor = None
        self.flavorData = None
        self.DirectoryEntry = SFNTDirectoryEntry
        self.file.seek(0)
        self.sfntVersion = self.file.read(4)
        self.file.seek(0)
        if self.sfntVersion == b'ttcf':
            header = readTTCHeader(self.file)
            numFonts = header.numFonts
            if not 0 <= fontNumber < numFonts:
                raise TTLibFileIsCollectionError('specify a font number between 0 and %d (inclusive)' % (numFonts - 1))
            self.numFonts = numFonts
            self.file.seek(header.offsetTable[fontNumber])
            data = self.file.read(sfntDirectorySize)
            if len(data) != sfntDirectorySize:
                raise TTLibError('Not a Font Collection (not enough data)')
            sstruct.unpack(sfntDirectoryFormat, data, self)
        elif self.sfntVersion == b'wOFF':
            self.flavor = 'woff'
            self.DirectoryEntry = WOFFDirectoryEntry
            data = self.file.read(woffDirectorySize)
            if len(data) != woffDirectorySize:
                raise TTLibError('Not a WOFF font (not enough data)')
            sstruct.unpack(woffDirectoryFormat, data, self)
        else:
            data = self.file.read(sfntDirectorySize)
            if len(data) != sfntDirectorySize:
                raise TTLibError('Not a TrueType or OpenType font (not enough data)')
            sstruct.unpack(sfntDirectoryFormat, data, self)
        self.sfntVersion = Tag(self.sfntVersion)
        if self.sfntVersion not in ('\x00\x01\x00\x00', 'OTTO', 'true'):
            raise TTLibError('Not a TrueType or OpenType font (bad sfntVersion)')
        tables = {}
        for i in range(self.numTables):
            entry = self.DirectoryEntry()
            entry.fromFile(self.file)
            tag = Tag(entry.tag)
            tables[tag] = entry
        self.tables = OrderedDict(sorted(tables.items(), key=lambda i: i[1].offset))
        if self.flavor == 'woff':
            self.flavorData = WOFFFlavorData(self)

    def has_key(self, tag):
        return tag in self.tables
    __contains__ = has_key

    def keys(self):
        return self.tables.keys()

    def __getitem__(self, tag):
        """Fetch the raw table data."""
        entry = self.tables[Tag(tag)]
        data = entry.loadData(self.file)
        if self.checkChecksums:
            if tag == 'head':
                checksum = calcChecksum(data[:8] + b'\x00\x00\x00\x00' + data[12:])
            else:
                checksum = calcChecksum(data)
            if self.checkChecksums > 1:
                assert checksum == entry.checkSum, "bad checksum for '%s' table" % tag
            elif checksum != entry.checkSum:
                log.warning("bad checksum for '%s' table", tag)
        return data

    def __delitem__(self, tag):
        del self.tables[Tag(tag)]

    def close(self):
        self.file.close()

    def __getstate__(self):
        if isinstance(self.file, BytesIO):
            return self.__dict__
        state = self.__dict__.copy()
        del state['file']
        state['_filename'] = self.file.name
        state['_filepos'] = self.file.tell()
        return state

    def __setstate__(self, state):
        if 'file' not in state:
            self.file = open(state.pop('_filename'), 'rb')
            self.file.seek(state.pop('_filepos'))
        self.__dict__.update(state)