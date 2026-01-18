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
class WOFF2Reader(SFNTReader):
    flavor = 'woff2'

    def __init__(self, file, checkChecksums=0, fontNumber=-1):
        if not haveBrotli:
            log.error('The WOFF2 decoder requires the Brotli Python extension, available at: https://github.com/google/brotli')
            raise ImportError('No module named brotli')
        self.file = file
        signature = Tag(self.file.read(4))
        if signature != b'wOF2':
            raise TTLibError('Not a WOFF2 font (bad signature)')
        self.file.seek(0)
        self.DirectoryEntry = WOFF2DirectoryEntry
        data = self.file.read(woff2DirectorySize)
        if len(data) != woff2DirectorySize:
            raise TTLibError('Not a WOFF2 font (not enough data)')
        sstruct.unpack(woff2DirectoryFormat, data, self)
        self.tables = OrderedDict()
        offset = 0
        for i in range(self.numTables):
            entry = self.DirectoryEntry()
            entry.fromFile(self.file)
            tag = Tag(entry.tag)
            self.tables[tag] = entry
            entry.offset = offset
            offset += entry.length
        totalUncompressedSize = offset
        compressedData = self.file.read(self.totalCompressedSize)
        decompressedData = brotli.decompress(compressedData)
        if len(decompressedData) != totalUncompressedSize:
            raise TTLibError('unexpected size for decompressed font data: expected %d, found %d' % (totalUncompressedSize, len(decompressedData)))
        self.transformBuffer = BytesIO(decompressedData)
        self.file.seek(0, 2)
        if self.length != self.file.tell():
            raise TTLibError("reported 'length' doesn't match the actual file size")
        self.flavorData = WOFF2FlavorData(self)
        self.ttFont = TTFont(recalcBBoxes=False, recalcTimestamp=False)

    def __getitem__(self, tag):
        """Fetch the raw table data. Reconstruct transformed tables."""
        entry = self.tables[Tag(tag)]
        if not hasattr(entry, 'data'):
            if entry.transformed:
                entry.data = self.reconstructTable(tag)
            else:
                entry.data = entry.loadData(self.transformBuffer)
        return entry.data

    def reconstructTable(self, tag):
        """Reconstruct table named 'tag' from transformed data."""
        entry = self.tables[Tag(tag)]
        rawData = entry.loadData(self.transformBuffer)
        if tag == 'glyf':
            padding = self.padding if hasattr(self, 'padding') else None
            data = self._reconstructGlyf(rawData, padding)
        elif tag == 'loca':
            data = self._reconstructLoca()
        elif tag == 'hmtx':
            data = self._reconstructHmtx(rawData)
        else:
            raise TTLibError("transform for table '%s' is unknown" % tag)
        return data

    def _reconstructGlyf(self, data, padding=None):
        """Return recostructed glyf table data, and set the corresponding loca's
        locations. Optionally pad glyph offsets to the specified number of bytes.
        """
        self.ttFont['loca'] = WOFF2LocaTable()
        glyfTable = self.ttFont['glyf'] = WOFF2GlyfTable()
        glyfTable.reconstruct(data, self.ttFont)
        if padding:
            glyfTable.padding = padding
        data = glyfTable.compile(self.ttFont)
        return data

    def _reconstructLoca(self):
        """Return reconstructed loca table data."""
        if 'loca' not in self.ttFont:
            self.tables['glyf'].data = self.reconstructTable('glyf')
        locaTable = self.ttFont['loca']
        data = locaTable.compile(self.ttFont)
        if len(data) != self.tables['loca'].origLength:
            raise TTLibError("reconstructed 'loca' table doesn't match original size: expected %d, found %d" % (self.tables['loca'].origLength, len(data)))
        return data

    def _reconstructHmtx(self, data):
        """Return reconstructed hmtx table data."""
        if 'glyf' in self.flavorData.transformedTables:
            tableDependencies = ('maxp', 'hhea', 'glyf')
        else:
            tableDependencies = ('maxp', 'head', 'hhea', 'loca', 'glyf')
        for tag in tableDependencies:
            self._decompileTable(tag)
        hmtxTable = self.ttFont['hmtx'] = WOFF2HmtxTable()
        hmtxTable.reconstruct(data, self.ttFont)
        data = hmtxTable.compile(self.ttFont)
        return data

    def _decompileTable(self, tag):
        """Decompile table data and store it inside self.ttFont."""
        data = self[tag]
        if self.ttFont.isLoaded(tag):
            return self.ttFont[tag]
        tableClass = getTableClass(tag)
        table = tableClass(tag)
        self.ttFont.tables[tag] = table
        table.decompile(data, self.ttFont)