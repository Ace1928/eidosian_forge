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
def _subset_glyphs(self, font):
    self.used_mark_sets = []
    for tag in self._sort_tables(font):
        clazz = ttLib.getTableClass(tag)
        if tag.strip() in self.options.no_subset_tables:
            log.info('%s subsetting not needed', tag)
        elif hasattr(clazz, 'subset_glyphs'):
            with timer("subset '%s'" % tag):
                table = font[tag]
                self.glyphs = self.glyphs_retained
                retain = table.subset_glyphs(self)
                del self.glyphs
            if not retain:
                log.info('%s subsetted to empty; dropped', tag)
                del font[tag]
            else:
                log.info('%s subsetted', tag)
        elif self.options.passthrough_tables:
            log.info("%s NOT subset; don't know how to subset", tag)
        else:
            log.warning("%s NOT subset; don't know how to subset; dropped", tag)
            del font[tag]
    with timer('subset GlyphOrder'):
        font.setGlyphOrder(self.new_glyph_order)