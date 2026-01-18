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
class AlternateSubst(FormatSwitchingBaseTable):

    def populateDefaults(self, propagator=None):
        if not hasattr(self, 'alternates'):
            self.alternates = {}

    def postRead(self, rawTable, font):
        alternates = {}
        if self.Format == 1:
            input = _getGlyphsFromCoverageTable(rawTable['Coverage'])
            alts = rawTable['AlternateSet']
            assert len(input) == len(alts)
            for inp, alt in zip(input, alts):
                alternates[inp] = alt.Alternate
        else:
            assert 0, 'unknown format: %s' % self.Format
        self.alternates = alternates
        del self.Format

    def preWrite(self, font):
        self.Format = 1
        alternates = getattr(self, 'alternates', None)
        if alternates is None:
            alternates = self.alternates = {}
        items = list(alternates.items())
        for i in range(len(items)):
            glyphName, set = items[i]
            items[i] = (font.getGlyphID(glyphName), glyphName, set)
        items.sort()
        cov = Coverage()
        cov.glyphs = [item[1] for item in items]
        alternates = []
        setList = [item[-1] for item in items]
        for set in setList:
            alts = AlternateSet()
            alts.Alternate = set
            alternates.append(alts)
        self.sortCoverageLast = 1
        return {'Coverage': cov, 'AlternateSet': alternates}

    def toXML2(self, xmlWriter, font):
        items = sorted(self.alternates.items())
        for glyphName, alternates in items:
            xmlWriter.begintag('AlternateSet', glyph=glyphName)
            xmlWriter.newline()
            for alt in alternates:
                xmlWriter.simpletag('Alternate', glyph=alt)
                xmlWriter.newline()
            xmlWriter.endtag('AlternateSet')
            xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        alternates = getattr(self, 'alternates', None)
        if alternates is None:
            alternates = {}
            self.alternates = alternates
        glyphName = attrs['glyph']
        set = []
        alternates[glyphName] = set
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            set.append(attrs['glyph'])