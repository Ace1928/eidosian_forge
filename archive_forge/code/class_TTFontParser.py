from struct import pack, unpack, error as structError
from reportlab.lib.utils import bytestr, isUnicode, char2int, isStr, isBytes
from reportlab.pdfbase import pdfmetrics, pdfdoc
from reportlab import rl_config
from reportlab.lib.rl_accel import hex32, add32, calcChecksum, instanceStringWidthTTF
from collections import namedtuple
from io import BytesIO
import os, time
from reportlab.rl_config import register_reset
class TTFontParser:
    """Basic TTF file parser"""
    ttfVersions = (65536, 1953658213, 1953784678)
    ttcVersions = (65536, 131072)
    fileKind = 'TTF'

    def __init__(self, file, validate=0, subfontIndex=0):
        """Loads and parses a TrueType font file.  file can be a filename or a
        file object.  If validate is set to a false values, skips checksum
        validation.  This can save time, especially if the font is large.
        """
        self.validate = validate
        self.readFile(file)
        isCollection = self.readHeader()
        if isCollection:
            self.readTTCHeader()
            self.getSubfont(subfontIndex)
        else:
            if self.validate:
                self.checksumFile()
            self.readTableDirectory()
            self.subfontNameX = b''

    def readTTCHeader(self):
        self.ttcVersion = self.read_ulong()
        self.fileKind = 'TTC'
        self.ttfVersions = self.ttfVersions[:-1]
        if self.ttcVersion not in self.ttcVersions:
            raise TTFError('"%s" is not a %s file: can\'t read version 0x%8.8x' % (self.filename, self.fileKind, self.ttcVersion))
        self.numSubfonts = self.read_ulong()
        self.subfontOffsets = []
        a = self.subfontOffsets.append
        for i in range(self.numSubfonts):
            a(self.read_ulong())

    def getSubfont(self, subfontIndex):
        if self.fileKind != 'TTC':
            raise TTFError('"%s" is not a TTC file: use this method' % (self.filename, self.fileKind))
        try:
            pos = self.subfontOffsets[subfontIndex]
        except IndexError:
            raise TTFError('TTC file "%s": bad subfontIndex %s not in [0,%d]' % (self.filename, subfontIndex, self.numSubfonts - 1))
        self.seek(pos)
        self.readHeader()
        self.readTableDirectory()
        self.subfontNameX = bytestr('-' + str(subfontIndex))

    def readTableDirectory(self):
        try:
            self.numTables = self.read_ushort()
            self.searchRange = self.read_ushort()
            self.entrySelector = self.read_ushort()
            self.rangeShift = self.read_ushort()
            self.table = {}
            self.tables = []
            for n in range(self.numTables):
                record = {}
                record['tag'] = self.read_tag()
                record['checksum'] = self.read_ulong()
                record['offset'] = self.read_ulong()
                record['length'] = self.read_ulong()
                self.tables.append(record)
                self.table[record['tag']] = record
        except:
            raise TTFError('Corrupt %s file "%s" cannot read Table Directory' % (self.fileKind, self.filename))
        if self.validate:
            self.checksumTables()

    def readHeader(self):
        """read the sfnt header at the current position"""
        try:
            self.version = version = self.read_ulong()
        except:
            raise TTFError('"%s" is not a %s file: can\'t read version' % (self.filename, self.fileKind))
        if version == 1330926671:
            raise TTFError('%s file "%s": postscript outlines are not supported' % (self.fileKind, self.filename))
        if version not in self.ttfVersions:
            raise TTFError('Not a recognized TrueType font: version=0x%8.8X' % version)
        return version == self.ttfVersions[-1]

    def readFile(self, f):
        if not hasattr(self, '_ttf_data'):
            if hasattr(f, 'read'):
                self.filename = getattr(f, 'name', '(ttf)')
                self._ttf_data = f.read()
            else:
                self.filename, f = TTFOpenFile(f)
                self._ttf_data = f.read()
                f.close()
        self._pos = 0

    def checksumTables(self):
        for t in self.tables:
            table = self.get_chunk(t['offset'], t['length'])
            checksum = calcChecksum(table)
            if t['tag'] == 'head':
                adjustment = unpack('>l', table[8:8 + 4])[0]
                checksum = add32(checksum, -adjustment)
            xchecksum = t['checksum']
            if xchecksum != checksum:
                raise TTFError('TTF file "%s": invalid checksum %s table: %s (expected %s)' % (self.filename, hex32(checksum), t['tag'], hex32(xchecksum)))

    def checksumFile(self):
        checksum = calcChecksum(self._ttf_data)
        if 2981146554 != checksum:
            raise TTFError('TTF file "%s": invalid checksum %s (expected 0xB1B0AFBA) len: %d &3: %d' % (self.filename, hex32(checksum), len(self._ttf_data), len(self._ttf_data) & 3))

    def get_table_pos(self, tag):
        """Returns the offset and size of a given TTF table."""
        offset = self.table[tag]['offset']
        length = self.table[tag]['length']
        return (offset, length)

    def seek(self, pos):
        """Moves read pointer to a given offset in file."""
        self._pos = pos

    def skip(self, delta):
        """Skip the given number of bytes."""
        self._pos = self._pos + delta

    def seek_table(self, tag, offset_in_table=0):
        """Moves read pointer to the given offset within a given table and
        returns absolute offset of that position in the file."""
        self._pos = self.get_table_pos(tag)[0] + offset_in_table
        return self._pos

    def read_tag(self):
        """Read a 4-character tag"""
        self._pos += 4
        return str(self._ttf_data[self._pos - 4:self._pos], 'utf8')

    def get_chunk(self, pos, length):
        """Return a chunk of raw data at given position"""
        return bytes(self._ttf_data[pos:pos + length])

    def read_uint8(self):
        self._pos += 1
        return int(self._ttf_data[self._pos - 1])

    def read_ushort(self):
        """Reads an unsigned short"""
        self._pos += 2
        return unpack('>H', self._ttf_data[self._pos - 2:self._pos])[0]

    def read_ulong(self):
        """Reads an unsigned long"""
        self._pos += 4
        return unpack('>L', self._ttf_data[self._pos - 4:self._pos])[0]

    def read_short(self):
        """Reads a signed short"""
        self._pos += 2
        try:
            return unpack('>h', self._ttf_data[self._pos - 2:self._pos])[0]
        except structError as error:
            raise TTFError(error)

    def get_ushort(self, pos):
        """Return an unsigned short at given position"""
        return unpack('>H', self._ttf_data[pos:pos + 2])[0]

    def get_ulong(self, pos):
        """Return an unsigned long at given position"""
        return unpack('>L', self._ttf_data[pos:pos + 4])[0]

    def get_table(self, tag):
        """Return the given TTF table"""
        pos, length = self.get_table_pos(tag)
        return self._ttf_data[pos:pos + length]