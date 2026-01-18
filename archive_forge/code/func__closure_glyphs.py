from fontTools import config
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables
from fontTools.ttLib.tables.otBase import USE_HARFBUZZ_REPACKER
from fontTools.otlLib.maxContextCalc import maxCtxFont
from fontTools.pens.basePen import NullPen
from fontTools.misc.loggingTools import Timer
from fontTools.misc.cliTools import makeOutputFileName
from fontTools.subset.util import _add_method, _uniq_sort
from fontTools.subset.cff import *
from fontTools.subset.svg import *
from fontTools.varLib import varStore  # for subset_varidxes
from fontTools.ttLib.tables._n_a_m_e import NameRecordVisitor
import sys
import struct
import array
import logging
from collections import Counter, defaultdict
from functools import reduce
from types import MethodType
def _closure_glyphs(self, font):
    realGlyphs = set(font.getGlyphOrder())
    self.orig_glyph_order = glyph_order = font.getGlyphOrder()
    self.glyphs_requested = set()
    self.glyphs_requested.update(self.glyph_names_requested)
    self.glyphs_requested.update((glyph_order[i] for i in self.glyph_ids_requested if i < len(glyph_order)))
    self.glyphs_missing = set()
    self.glyphs_missing.update(self.glyphs_requested.difference(realGlyphs))
    self.glyphs_missing.update((i for i in self.glyph_ids_requested if i >= len(glyph_order)))
    if self.glyphs_missing:
        log.info('Missing requested glyphs: %s', self.glyphs_missing)
        if not self.options.ignore_missing_glyphs:
            raise self.MissingGlyphsSubsettingError(self.glyphs_missing)
    self.glyphs = self.glyphs_requested.copy()
    self.unicodes_missing = set()
    if 'cmap' in font:
        with timer("close glyph list over 'cmap'"):
            font['cmap'].closure_glyphs(self)
            self.glyphs.intersection_update(realGlyphs)
    self.glyphs_cmaped = frozenset(self.glyphs)
    if self.unicodes_missing:
        missing = ['U+%04X' % u for u in self.unicodes_missing]
        log.info('Missing glyphs for requested Unicodes: %s', missing)
        if not self.options.ignore_missing_unicodes:
            raise self.MissingUnicodesSubsettingError(missing)
        del missing
    if self.options.notdef_glyph:
        if 'glyf' in font:
            self.glyphs.add(font.getGlyphName(0))
            log.info('Added gid0 to subset')
        else:
            self.glyphs.add('.notdef')
            log.info('Added .notdef to subset')
    if self.options.recommended_glyphs:
        if 'glyf' in font:
            for i in range(min(4, len(font.getGlyphOrder()))):
                self.glyphs.add(font.getGlyphName(i))
            log.info('Added first four glyphs to subset')
    if self.options.layout_closure and 'GSUB' in font:
        with timer("close glyph list over 'GSUB'"):
            log.info("Closing glyph list over 'GSUB': %d glyphs before", len(self.glyphs))
            log.glyphs(self.glyphs, font=font)
            font['GSUB'].closure_glyphs(self)
            self.glyphs.intersection_update(realGlyphs)
            log.info("Closed glyph list over 'GSUB': %d glyphs after", len(self.glyphs))
            log.glyphs(self.glyphs, font=font)
    self.glyphs_gsubed = frozenset(self.glyphs)
    if 'MATH' in font:
        with timer("close glyph list over 'MATH'"):
            log.info("Closing glyph list over 'MATH': %d glyphs before", len(self.glyphs))
            log.glyphs(self.glyphs, font=font)
            font['MATH'].closure_glyphs(self)
            self.glyphs.intersection_update(realGlyphs)
            log.info("Closed glyph list over 'MATH': %d glyphs after", len(self.glyphs))
            log.glyphs(self.glyphs, font=font)
    self.glyphs_mathed = frozenset(self.glyphs)
    for table in ('COLR', 'bsln'):
        if table in font:
            with timer("close glyph list over '%s'" % table):
                log.info("Closing glyph list over '%s': %d glyphs before", table, len(self.glyphs))
                log.glyphs(self.glyphs, font=font)
                font[table].closure_glyphs(self)
                self.glyphs.intersection_update(realGlyphs)
                log.info("Closed glyph list over '%s': %d glyphs after", table, len(self.glyphs))
                log.glyphs(self.glyphs, font=font)
        setattr(self, f'glyphs_{table.lower()}ed', frozenset(self.glyphs))
    if 'glyf' in font:
        with timer("close glyph list over 'glyf'"):
            log.info("Closing glyph list over 'glyf': %d glyphs before", len(self.glyphs))
            log.glyphs(self.glyphs, font=font)
            font['glyf'].closure_glyphs(self)
            self.glyphs.intersection_update(realGlyphs)
            log.info("Closed glyph list over 'glyf': %d glyphs after", len(self.glyphs))
            log.glyphs(self.glyphs, font=font)
    self.glyphs_glyfed = frozenset(self.glyphs)
    if 'CFF ' in font:
        with timer("close glyph list over 'CFF '"):
            log.info("Closing glyph list over 'CFF ': %d glyphs before", len(self.glyphs))
            log.glyphs(self.glyphs, font=font)
            font['CFF '].closure_glyphs(self)
            self.glyphs.intersection_update(realGlyphs)
            log.info("Closed glyph list over 'CFF ': %d glyphs after", len(self.glyphs))
            log.glyphs(self.glyphs, font=font)
    self.glyphs_cffed = frozenset(self.glyphs)
    self.glyphs_retained = frozenset(self.glyphs)
    order = font.getReverseGlyphMap()
    self.reverseOrigGlyphMap = {g: order[g] for g in self.glyphs_retained}
    self.last_retained_order = max(self.reverseOrigGlyphMap.values())
    self.last_retained_glyph = font.getGlyphOrder()[self.last_retained_order]
    self.glyphs_emptied = frozenset()
    if self.options.retain_gids:
        self.glyphs_emptied = {g for g in realGlyphs - self.glyphs_retained if order[g] <= self.last_retained_order}
    self.reverseEmptiedGlyphMap = {g: order[g] for g in self.glyphs_emptied}
    if not self.options.retain_gids:
        new_glyph_order = [g for g in glyph_order if g in self.glyphs_retained]
    else:
        new_glyph_order = [g for g in glyph_order if font.getGlyphID(g) <= self.last_retained_order]
    self.new_glyph_order = new_glyph_order
    self.glyph_index_map = {order[new_glyph_order[i]]: i for i in range(len(new_glyph_order))}
    log.info('Retaining %d glyphs', len(self.glyphs_retained))
    del self.glyphs