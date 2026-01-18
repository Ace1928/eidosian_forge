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
class SingleSubst(FormatSwitchingBaseTable):

    def populateDefaults(self, propagator=None):
        if not hasattr(self, 'mapping'):
            self.mapping = {}

    def postRead(self, rawTable, font):
        mapping = {}
        input = _getGlyphsFromCoverageTable(rawTable['Coverage'])
        if self.Format == 1:
            delta = rawTable['DeltaGlyphID']
            inputGIDS = font.getGlyphIDMany(input)
            outGIDS = [(glyphID + delta) % 65536 for glyphID in inputGIDS]
            outNames = font.getGlyphNameMany(outGIDS)
            for inp, out in zip(input, outNames):
                mapping[inp] = out
        elif self.Format == 2:
            assert len(input) == rawTable['GlyphCount'], 'invalid SingleSubstFormat2 table'
            subst = rawTable['Substitute']
            for inp, sub in zip(input, subst):
                mapping[inp] = sub
        else:
            assert 0, 'unknown format: %s' % self.Format
        self.mapping = mapping
        del self.Format

    def preWrite(self, font):
        mapping = getattr(self, 'mapping', None)
        if mapping is None:
            mapping = self.mapping = {}
        items = list(mapping.items())
        getGlyphID = font.getGlyphID
        gidItems = [(getGlyphID(a), getGlyphID(b)) for a, b in items]
        sortableItems = sorted(zip(gidItems, items))
        format = 2
        delta = None
        for inID, outID in gidItems:
            if delta is None:
                delta = (outID - inID) % 65536
            if (inID + delta) % 65536 != outID:
                break
        else:
            if delta is None:
                format = 2
            else:
                format = 1
        rawTable = {}
        self.Format = format
        cov = Coverage()
        input = [item[1][0] for item in sortableItems]
        subst = [item[1][1] for item in sortableItems]
        cov.glyphs = input
        rawTable['Coverage'] = cov
        if format == 1:
            assert delta is not None
            rawTable['DeltaGlyphID'] = delta
        else:
            rawTable['Substitute'] = subst
        return rawTable

    def toXML2(self, xmlWriter, font):
        items = sorted(self.mapping.items())
        for inGlyph, outGlyph in items:
            xmlWriter.simpletag('Substitution', [('in', inGlyph), ('out', outGlyph)])
            xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        mapping = getattr(self, 'mapping', None)
        if mapping is None:
            mapping = {}
            self.mapping = mapping
        mapping[attrs['in']] = attrs['out']