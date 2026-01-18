from fontTools.misc import sstruct
from fontTools.misc.fixedTools import floatToFixedToStr, strToFixedToFloat
from fontTools.misc.textTools import safeEval, num2binary, binary2num
from fontTools.misc.timeTools import (
from fontTools.misc.timeTools import epoch_diff as mac_epoch_diff  # For backward compat
from fontTools.misc.arrayTools import intRect, unionRect
from . import DefaultTable
import logging
class table__h_e_a_d(DefaultTable.DefaultTable):
    dependencies = ['maxp', 'loca', 'CFF ', 'CFF2']

    def decompile(self, data, ttFont):
        dummy, rest = sstruct.unpack2(headFormat, data, self)
        if rest:
            log.warning("extra bytes at the end of 'head' table")
            assert rest == b'\x00\x00'
        for stamp in ('created', 'modified'):
            value = getattr(self, stamp)
            if value > 4294967295:
                log.warning("'%s' timestamp out of range; ignoring top bytes", stamp)
                value &= 4294967295
                setattr(self, stamp, value)
            if value < 2082840000:
                log.warning("'%s' timestamp seems very low; regarding as unix timestamp", stamp)
                value += 2082840000
                setattr(self, stamp, value)

    def compile(self, ttFont):
        if ttFont.recalcBBoxes:
            if 'CFF ' in ttFont:
                topDict = ttFont['CFF '].cff.topDictIndex[0]
                self.xMin, self.yMin, self.xMax, self.yMax = intRect(topDict.FontBBox)
            elif 'CFF2' in ttFont:
                topDict = ttFont['CFF2'].cff.topDictIndex[0]
                charStrings = topDict.CharStrings
                fontBBox = None
                for charString in charStrings.values():
                    bounds = charString.calcBounds(charStrings)
                    if bounds is not None:
                        if fontBBox is not None:
                            fontBBox = unionRect(fontBBox, bounds)
                        else:
                            fontBBox = bounds
                if fontBBox is not None:
                    self.xMin, self.yMin, self.xMax, self.yMax = intRect(fontBBox)
        if ttFont.recalcTimestamp:
            self.modified = timestampNow()
        data = sstruct.pack(headFormat, self)
        return data

    def toXML(self, writer, ttFont):
        writer.comment('Most of this table will be recalculated by the compiler')
        writer.newline()
        _, names, fixes = sstruct.getformat(headFormat)
        for name in names:
            value = getattr(self, name)
            if name in fixes:
                value = floatToFixedToStr(value, precisionBits=fixes[name])
            elif name in ('created', 'modified'):
                value = timestampToString(value)
            elif name in ('magicNumber', 'checkSumAdjustment'):
                if value < 0:
                    value = value + 4294967296
                value = hex(value)
                if value[-1:] == 'L':
                    value = value[:-1]
            elif name in ('macStyle', 'flags'):
                value = num2binary(value, 16)
            writer.simpletag(name, value=value)
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        value = attrs['value']
        fixes = sstruct.getformat(headFormat)[2]
        if name in fixes:
            value = strToFixedToFloat(value, precisionBits=fixes[name])
        elif name in ('created', 'modified'):
            value = timestampFromString(value)
        elif name in ('macStyle', 'flags'):
            value = binary2num(value)
        else:
            value = safeEval(value)
        setattr(self, name, value)