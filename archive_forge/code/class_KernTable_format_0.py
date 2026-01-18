from fontTools.ttLib import getSearchRange
from fontTools.misc.textTools import safeEval, readHex
from fontTools.misc.fixedTools import fixedToFloat as fi2fl, floatToFixed as fl2fi
from . import DefaultTable
import struct
import sys
import array
import logging
class KernTable_format_0(object):
    version = format = 0

    def __init__(self, apple=False):
        self.apple = apple

    def decompile(self, data, ttFont):
        if not self.apple:
            version, length, subtableFormat, coverage = struct.unpack('>HHBB', data[:6])
            if version != 0:
                from fontTools.ttLib import TTLibError
                raise TTLibError('unsupported kern subtable version: %d' % version)
            tupleIndex = None
            data = data[6:]
        else:
            length, coverage, subtableFormat, tupleIndex = struct.unpack('>LBBH', data[:8])
            data = data[8:]
        assert self.format == subtableFormat, 'unsupported format'
        self.coverage = coverage
        self.tupleIndex = tupleIndex
        self.kernTable = kernTable = {}
        nPairs, searchRange, entrySelector, rangeShift = struct.unpack('>HHHH', data[:8])
        data = data[8:]
        datas = array.array('H', data[:6 * nPairs])
        if sys.byteorder != 'big':
            datas.byteswap()
        it = iter(datas)
        glyphOrder = ttFont.getGlyphOrder()
        for k in range(nPairs):
            left, right, value = (next(it), next(it), next(it))
            if value >= 32768:
                value -= 65536
            try:
                kernTable[glyphOrder[left], glyphOrder[right]] = value
            except IndexError:
                kernTable[ttFont.getGlyphName(left), ttFont.getGlyphName(right)] = value
        if len(data) > 6 * nPairs + 4:
            log.warning("excess data in 'kern' subtable: %d bytes", len(data) - 6 * nPairs)

    def compile(self, ttFont):
        nPairs = min(len(self.kernTable), 65535)
        searchRange, entrySelector, rangeShift = getSearchRange(nPairs, 6)
        searchRange &= 65535
        entrySelector = min(entrySelector, 65535)
        rangeShift = min(rangeShift, 65535)
        data = struct.pack('>HHHH', nPairs, searchRange, entrySelector, rangeShift)
        try:
            reverseOrder = ttFont.getReverseGlyphMap()
            kernTable = sorted(((reverseOrder[left], reverseOrder[right], value) for (left, right), value in self.kernTable.items()))
        except KeyError:
            getGlyphID = ttFont.getGlyphID
            kernTable = sorted(((getGlyphID(left), getGlyphID(right), value) for (left, right), value in self.kernTable.items()))
        for left, right, value in kernTable:
            data = data + struct.pack('>HHh', left, right, value)
        if not self.apple:
            version = 0
            length = len(data) + 6
            if length >= 65536:
                log.warning('"kern" subtable overflow, truncating length value while preserving pairs.')
                length &= 65535
            header = struct.pack('>HHBB', version, length, self.format, self.coverage)
        else:
            if self.tupleIndex is None:
                log.warning("'tupleIndex' is None; default to 0")
                self.tupleIndex = 0
            length = len(data) + 8
            header = struct.pack('>LBBH', length, self.coverage, self.format, self.tupleIndex)
        return header + data

    def toXML(self, writer, ttFont):
        attrs = dict(coverage=self.coverage, format=self.format)
        if self.apple:
            if self.tupleIndex is None:
                log.warning("'tupleIndex' is None; default to 0")
                attrs['tupleIndex'] = 0
            else:
                attrs['tupleIndex'] = self.tupleIndex
        writer.begintag('kernsubtable', **attrs)
        writer.newline()
        items = sorted(self.kernTable.items())
        for (left, right), value in items:
            writer.simpletag('pair', [('l', left), ('r', right), ('v', value)])
            writer.newline()
        writer.endtag('kernsubtable')
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.coverage = safeEval(attrs['coverage'])
        subtableFormat = safeEval(attrs['format'])
        if self.apple:
            if 'tupleIndex' in attrs:
                self.tupleIndex = safeEval(attrs['tupleIndex'])
            else:
                log.warning("Apple kern subtable is missing 'tupleIndex' attribute")
                self.tupleIndex = None
        else:
            self.tupleIndex = None
        assert subtableFormat == self.format, 'unsupported format'
        if not hasattr(self, 'kernTable'):
            self.kernTable = {}
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            self.kernTable[attrs['l'], attrs['r']] = safeEval(attrs['v'])

    def __getitem__(self, pair):
        return self.kernTable[pair]

    def __setitem__(self, pair, value):
        self.kernTable[pair] = value

    def __delitem__(self, pair):
        del self.kernTable[pair]