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
class LigatureSubst(FormatSwitchingBaseTable):

    def populateDefaults(self, propagator=None):
        if not hasattr(self, 'ligatures'):
            self.ligatures = {}

    def postRead(self, rawTable, font):
        ligatures = {}
        if self.Format == 1:
            input = _getGlyphsFromCoverageTable(rawTable['Coverage'])
            ligSets = rawTable['LigatureSet']
            assert len(input) == len(ligSets)
            for i in range(len(input)):
                ligatures[input[i]] = ligSets[i].Ligature
        else:
            assert 0, 'unknown format: %s' % self.Format
        self.ligatures = ligatures
        del self.Format

    @staticmethod
    def _getLigatureSortKey(components):
        return -len(components)

    def preWrite(self, font):
        self.Format = 1
        ligatures = getattr(self, 'ligatures', None)
        if ligatures is None:
            ligatures = self.ligatures = {}
        if ligatures and isinstance(next(iter(ligatures)), tuple):
            newLigatures = dict()
            for comps in sorted(ligatures.keys(), key=self._getLigatureSortKey):
                ligature = Ligature()
                ligature.Component = comps[1:]
                ligature.CompCount = len(comps)
                ligature.LigGlyph = ligatures[comps]
                newLigatures.setdefault(comps[0], []).append(ligature)
            ligatures = newLigatures
        items = list(ligatures.items())
        for i in range(len(items)):
            glyphName, set = items[i]
            items[i] = (font.getGlyphID(glyphName), glyphName, set)
        items.sort()
        cov = Coverage()
        cov.glyphs = [item[1] for item in items]
        ligSets = []
        setList = [item[-1] for item in items]
        for set in setList:
            ligSet = LigatureSet()
            ligs = ligSet.Ligature = []
            for lig in set:
                ligs.append(lig)
            ligSets.append(ligSet)
        self.sortCoverageLast = 1
        return {'Coverage': cov, 'LigatureSet': ligSets}

    def toXML2(self, xmlWriter, font):
        items = sorted(self.ligatures.items())
        for glyphName, ligSets in items:
            xmlWriter.begintag('LigatureSet', glyph=glyphName)
            xmlWriter.newline()
            for lig in ligSets:
                xmlWriter.simpletag('Ligature', glyph=lig.LigGlyph, components=','.join(lig.Component))
                xmlWriter.newline()
            xmlWriter.endtag('LigatureSet')
            xmlWriter.newline()

    def fromXML(self, name, attrs, content, font):
        ligatures = getattr(self, 'ligatures', None)
        if ligatures is None:
            ligatures = {}
            self.ligatures = ligatures
        glyphName = attrs['glyph']
        ligs = []
        ligatures[glyphName] = ligs
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            lig = Ligature()
            lig.LigGlyph = attrs['glyph']
            components = attrs['components']
            lig.Component = components.split(',') if components else []
            lig.CompCount = len(lig.Component)
            ligs.append(lig)