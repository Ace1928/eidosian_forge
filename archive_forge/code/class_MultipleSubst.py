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
class MultipleSubst(FormatSwitchingBaseTable):

    def populateDefaults(self, propagator=None):
        if not hasattr(self, 'mapping'):
            self.mapping = {}

    def postRead(self, rawTable, font):
        mapping = {}
        if self.Format == 1:
            glyphs = _getGlyphsFromCoverageTable(rawTable['Coverage'])
            subst = [s.Substitute for s in rawTable['Sequence']]
            mapping = dict(zip(glyphs, subst))
        else:
            assert 0, 'unknown format: %s' % self.Format
        self.mapping = mapping
        del self.Format

    def preWrite(self, font):
        mapping = getattr(self, 'mapping', None)
        if mapping is None:
            mapping = self.mapping = {}
        cov = Coverage()
        cov.glyphs = sorted(list(mapping.keys()), key=font.getGlyphID)
        self.Format = 1
        rawTable = {'Coverage': cov, 'Sequence': [self.makeSequence_(mapping[glyph]) for glyph in cov.glyphs]}
        return rawTable

    def toXML2(self, xmlWriter, font):
        items = sorted(self.mapping.items())
        for inGlyph, outGlyphs in items:
            out = ','.join(outGlyphs)
            xmlWriter.simpletag('Substitution', [('in', inGlyph), ('out', out)])
            xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        mapping = getattr(self, 'mapping', None)
        if mapping is None:
            mapping = {}
            self.mapping = mapping
        if name == 'Coverage':
            self.old_coverage_ = []
            for element in content:
                if not isinstance(element, tuple):
                    continue
                element_name, element_attrs, _ = element
                if element_name == 'Glyph':
                    self.old_coverage_.append(element_attrs['value'])
            return
        if name == 'Sequence':
            index = int(attrs.get('index', len(mapping)))
            glyph = self.old_coverage_[index]
            glyph_mapping = mapping[glyph] = []
            for element in content:
                if not isinstance(element, tuple):
                    continue
                element_name, element_attrs, _ = element
                if element_name == 'Substitute':
                    glyph_mapping.append(element_attrs['value'])
            return
        outGlyphs = attrs['out'].split(',') if attrs['out'] else []
        mapping[attrs['in']] = [g.strip() for g in outGlyphs]

    @staticmethod
    def makeSequence_(g):
        seq = Sequence()
        seq.Substitute = g
        return seq