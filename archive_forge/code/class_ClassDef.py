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
class ClassDef(FormatSwitchingBaseTable):

    def populateDefaults(self, propagator=None):
        if not hasattr(self, 'classDefs'):
            self.classDefs = {}

    def postRead(self, rawTable, font):
        classDefs = {}
        if self.Format == 1:
            start = rawTable['StartGlyph']
            classList = rawTable['ClassValueArray']
            startID = font.getGlyphID(start)
            endID = startID + len(classList)
            glyphNames = font.getGlyphNameMany(range(startID, endID))
            for glyphName, cls in zip(glyphNames, classList):
                if cls:
                    classDefs[glyphName] = cls
        elif self.Format == 2:
            records = rawTable['ClassRangeRecord']
            for rec in records:
                cls = rec.Class
                if not cls:
                    continue
                start = rec.Start
                end = rec.End
                startID = font.getGlyphID(start)
                endID = font.getGlyphID(end) + 1
                glyphNames = font.getGlyphNameMany(range(startID, endID))
                for glyphName in glyphNames:
                    classDefs[glyphName] = cls
        else:
            log.warning('Unknown ClassDef format: %s', self.Format)
        self.classDefs = classDefs
        del self.Format

    def _getClassRanges(self, font):
        classDefs = getattr(self, 'classDefs', None)
        if classDefs is None:
            self.classDefs = {}
            return
        getGlyphID = font.getGlyphID
        items = []
        for glyphName, cls in classDefs.items():
            if not cls:
                continue
            items.append((getGlyphID(glyphName), glyphName, cls))
        if items:
            items.sort()
            last, lastName, lastCls = items[0]
            ranges = [[lastCls, last, lastName]]
            for glyphID, glyphName, cls in items[1:]:
                if glyphID != last + 1 or cls != lastCls:
                    ranges[-1].extend([last, lastName])
                    ranges.append([cls, glyphID, glyphName])
                last = glyphID
                lastName = glyphName
                lastCls = cls
            ranges[-1].extend([last, lastName])
            return ranges

    def preWrite(self, font):
        format = 2
        rawTable = {'ClassRangeRecord': []}
        ranges = self._getClassRanges(font)
        if ranges:
            startGlyph = ranges[0][1]
            endGlyph = ranges[-1][3]
            glyphCount = endGlyph - startGlyph + 1
            if len(ranges) * 3 < glyphCount + 1:
                for i in range(len(ranges)):
                    cls, start, startName, end, endName = ranges[i]
                    rec = ClassRangeRecord()
                    rec.Start = startName
                    rec.End = endName
                    rec.Class = cls
                    ranges[i] = rec
                format = 2
                rawTable = {'ClassRangeRecord': ranges}
            else:
                startGlyphName = ranges[0][2]
                classes = [0] * glyphCount
                for cls, start, startName, end, endName in ranges:
                    for g in range(start - startGlyph, end - startGlyph + 1):
                        classes[g] = cls
                format = 1
                rawTable = {'StartGlyph': startGlyphName, 'ClassValueArray': classes}
        self.Format = format
        return rawTable

    def toXML2(self, xmlWriter, font):
        items = sorted(self.classDefs.items())
        for glyphName, cls in items:
            xmlWriter.simpletag('ClassDef', [('glyph', glyphName), ('class', cls)])
            xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        classDefs = getattr(self, 'classDefs', None)
        if classDefs is None:
            classDefs = {}
            self.classDefs = classDefs
        classDefs[attrs['glyph']] = int(attrs['class'])