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
class VarIdxMap(BaseTable):

    def populateDefaults(self, propagator=None):
        if not hasattr(self, 'mapping'):
            self.mapping = {}

    def postRead(self, rawTable, font):
        assert rawTable['EntryFormat'] & 65472 == 0
        glyphOrder = font.getGlyphOrder()
        mapList = rawTable['mapping']
        mapList.extend([mapList[-1]] * (len(glyphOrder) - len(mapList)))
        self.mapping = dict(zip(glyphOrder, mapList))

    def preWrite(self, font):
        mapping = getattr(self, 'mapping', None)
        if mapping is None:
            mapping = self.mapping = {}
        glyphOrder = font.getGlyphOrder()
        mapping = [mapping[g] for g in glyphOrder]
        while len(mapping) > 1 and mapping[-2] == mapping[-1]:
            del mapping[-1]
        rawTable = {'mapping': mapping}
        rawTable['MappingCount'] = len(mapping)
        rawTable['EntryFormat'] = DeltaSetIndexMap.getEntryFormat(mapping)
        return rawTable

    def toXML2(self, xmlWriter, font):
        for glyph, value in sorted(getattr(self, 'mapping', {}).items()):
            attrs = (('glyph', glyph), ('outer', value >> 16), ('inner', value & 65535))
            xmlWriter.simpletag('Map', attrs)
            xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        mapping = getattr(self, 'mapping', None)
        if mapping is None:
            mapping = {}
            self.mapping = mapping
        try:
            glyph = attrs['glyph']
        except:
            glyph = font.getGlyphOrder()[attrs['index']]
        outer = safeEval(attrs['outer'])
        inner = safeEval(attrs['inner'])
        assert inner <= 65535
        mapping[glyph] = outer << 16 | inner