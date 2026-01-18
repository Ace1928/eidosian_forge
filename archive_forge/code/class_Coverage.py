import copy
from enum import IntEnum
from functools import reduce
from math import radians
import itertools
from collections import defaultdict, namedtuple
from fontTools.ttLib.tables.otTraverse import dfs_base_table
from fontTools.misc.arrayTools import quantizeRect
from fontTools.misc.roundTools import otRound
from fontTools.misc.transform import Transform, Identity
from fontTools.misc.textTools import bytesjoin, pad, safeEval
from fontTools.pens.boundsPen import ControlBoundsPen
from fontTools.pens.transformPen import TransformPen
from .otBase import (
from fontTools.feaLib.lookupDebugInfo import LookupDebugInfo, LOOKUP_DEBUG_INFO_KEY
import logging
import struct
from typing import TYPE_CHECKING, Iterator, List, Optional, Set
class Coverage(FormatSwitchingBaseTable):

    def populateDefaults(self, propagator=None):
        if not hasattr(self, 'glyphs'):
            self.glyphs = []

    def postRead(self, rawTable, font):
        if self.Format == 1:
            self.glyphs = rawTable['GlyphArray']
        elif self.Format == 2:
            glyphs = self.glyphs = []
            ranges = rawTable['RangeRecord']
            sorted_ranges = sorted(ranges, key=lambda a: a.StartCoverageIndex)
            if ranges != sorted_ranges:
                log.warning('GSUB/GPOS Coverage is not sorted by glyph ids.')
                ranges = sorted_ranges
            del sorted_ranges
            for r in ranges:
                start = r.Start
                end = r.End
                startID = font.getGlyphID(start)
                endID = font.getGlyphID(end) + 1
                glyphs.extend(font.getGlyphNameMany(range(startID, endID)))
        else:
            self.glyphs = []
            log.warning('Unknown Coverage format: %s', self.Format)
        del self.Format

    def preWrite(self, font):
        glyphs = getattr(self, 'glyphs', None)
        if glyphs is None:
            glyphs = self.glyphs = []
        format = 1
        rawTable = {'GlyphArray': glyphs}
        if glyphs:
            glyphIDs = font.getGlyphIDMany(glyphs)
            brokenOrder = sorted(glyphIDs) != glyphIDs
            last = glyphIDs[0]
            ranges = [[last]]
            for glyphID in glyphIDs[1:]:
                if glyphID != last + 1:
                    ranges[-1].append(last)
                    ranges.append([glyphID])
                last = glyphID
            ranges[-1].append(last)
            if brokenOrder or len(ranges) * 3 < len(glyphs):
                index = 0
                for i in range(len(ranges)):
                    start, end = ranges[i]
                    r = RangeRecord()
                    r.StartID = start
                    r.Start = font.getGlyphName(start)
                    r.End = font.getGlyphName(end)
                    r.StartCoverageIndex = index
                    ranges[i] = r
                    index = index + end - start + 1
                if brokenOrder:
                    log.warning('GSUB/GPOS Coverage is not sorted by glyph ids.')
                    ranges.sort(key=lambda a: a.StartID)
                for r in ranges:
                    del r.StartID
                format = 2
                rawTable = {'RangeRecord': ranges}
        self.Format = format
        return rawTable

    def toXML2(self, xmlWriter, font):
        for glyphName in getattr(self, 'glyphs', []):
            xmlWriter.simpletag('Glyph', value=glyphName)
            xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        glyphs = getattr(self, 'glyphs', None)
        if glyphs is None:
            glyphs = []
            self.glyphs = glyphs
        glyphs.append(attrs['value'])