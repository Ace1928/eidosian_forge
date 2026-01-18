from collections import namedtuple, OrderedDict
import os
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import (
from fontTools.ttLib.tables import otBase
from fontTools.feaLib.ast import STATNameStatement
from fontTools.otlLib.optimize.gpos import (
from fontTools.otlLib.error import OpenTypeLibError
from functools import reduce
import logging
import copy
def _buildMathGlyphInfo(glyphMap, italicsCorrections, topAccentAttachments, extendedShapes, mathKerns):
    if not any([extendedShapes, italicsCorrections, topAccentAttachments, mathKerns]):
        return None
    info = ot.MathGlyphInfo()
    info.populateDefaults()
    if italicsCorrections:
        coverage = buildCoverage(italicsCorrections.keys(), glyphMap)
        info.MathItalicsCorrectionInfo = ot.MathItalicsCorrectionInfo()
        info.MathItalicsCorrectionInfo.Coverage = coverage
        info.MathItalicsCorrectionInfo.ItalicsCorrectionCount = len(coverage.glyphs)
        info.MathItalicsCorrectionInfo.ItalicsCorrection = [_mathValueRecord(italicsCorrections[n]) for n in coverage.glyphs]
    if topAccentAttachments:
        coverage = buildCoverage(topAccentAttachments.keys(), glyphMap)
        info.MathTopAccentAttachment = ot.MathTopAccentAttachment()
        info.MathTopAccentAttachment.TopAccentCoverage = coverage
        info.MathTopAccentAttachment.TopAccentAttachmentCount = len(coverage.glyphs)
        info.MathTopAccentAttachment.TopAccentAttachment = [_mathValueRecord(topAccentAttachments[n]) for n in coverage.glyphs]
    if extendedShapes:
        info.ExtendedShapeCoverage = buildCoverage(extendedShapes, glyphMap)
    if mathKerns:
        coverage = buildCoverage(mathKerns.keys(), glyphMap)
        info.MathKernInfo = ot.MathKernInfo()
        info.MathKernInfo.MathKernCoverage = coverage
        info.MathKernInfo.MathKernCount = len(coverage.glyphs)
        info.MathKernInfo.MathKernInfoRecords = []
        for glyph in coverage.glyphs:
            record = ot.MathKernInfoRecord()
            for side in {'TopRight', 'TopLeft', 'BottomRight', 'BottomLeft'}:
                if side in mathKerns[glyph]:
                    correctionHeights, kernValues = mathKerns[glyph][side]
                    assert len(correctionHeights) == len(kernValues) - 1
                    kern = ot.MathKern()
                    kern.HeightCount = len(correctionHeights)
                    kern.CorrectionHeight = [_mathValueRecord(h) for h in correctionHeights]
                    kern.KernValue = [_mathValueRecord(v) for v in kernValues]
                    setattr(record, f'{side}MathKern', kern)
            info.MathKernInfo.MathKernInfoRecords.append(record)
    return info