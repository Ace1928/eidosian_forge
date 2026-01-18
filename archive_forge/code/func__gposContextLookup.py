import logging
import re
from io import StringIO
from fontTools.feaLib import ast
from fontTools.ttLib import TTFont, TTLibError
from fontTools.voltLib import ast as VAst
from fontTools.voltLib.parser import Parser as VoltParser
def _gposContextLookup(self, lookup, prefix, suffix, ignore, fealookup, targetlookup):
    statements = fealookup.statements
    assert not lookup.reversal
    pos = lookup.pos
    if isinstance(pos, VAst.PositionAdjustPairDefinition):
        for (idx1, idx2), (pos1, pos2) in pos.adjust_pair.items():
            glyphs1 = self._coverage(pos.coverages_1[idx1 - 1])
            glyphs2 = self._coverage(pos.coverages_2[idx2 - 1])
            assert len(glyphs1) == 1
            assert len(glyphs2) == 1
            glyphs = (glyphs1[0], glyphs2[0])
            if ignore:
                statement = ast.IgnorePosStatement([(prefix, glyphs, suffix)])
            else:
                lookups = (targetlookup, targetlookup)
                statement = ast.ChainContextPosStatement(prefix, glyphs, suffix, lookups)
            statements.append(statement)
    elif isinstance(pos, VAst.PositionAdjustSingleDefinition):
        glyphs = [ast.GlyphClass()]
        for a, b in pos.adjust_single:
            glyph = self._coverage(a)
            glyphs[0].extend(glyph)
        if ignore:
            statement = ast.IgnorePosStatement([(prefix, glyphs, suffix)])
        else:
            statement = ast.ChainContextPosStatement(prefix, glyphs, suffix, [targetlookup])
        statements.append(statement)
    elif isinstance(pos, VAst.PositionAttachDefinition):
        glyphs = [ast.GlyphClass()]
        for coverage, _ in pos.coverage_to:
            glyphs[0].extend(self._coverage(coverage))
        if ignore:
            statement = ast.IgnorePosStatement([(prefix, glyphs, suffix)])
        else:
            statement = ast.ChainContextPosStatement(prefix, glyphs, suffix, [targetlookup])
        statements.append(statement)
    else:
        raise NotImplementedError(pos)