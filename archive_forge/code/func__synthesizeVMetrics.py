from collections import namedtuple
from fontTools.misc import sstruct
from fontTools import ttLib
from fontTools import version
from fontTools.misc.transform import DecomposedTransform
from fontTools.misc.textTools import tostr, safeEval, pad
from fontTools.misc.arrayTools import updateBounds, pointInRect
from fontTools.misc.bezierTools import calcQuadraticBounds
from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.vector import Vector
from numbers import Number
from . import DefaultTable
from . import ttProgram
import sys
import struct
import array
import logging
import math
import os
from fontTools.misc import xmlWriter
from fontTools.misc.filenames import userNameToFileName
from fontTools.misc.loggingTools import deprecateFunction
from enum import IntFlag
from functools import partial
from types import SimpleNamespace
from typing import Set
def _synthesizeVMetrics(self, glyphName, ttFont, defaultVerticalOrigin):
    """This method is wrong and deprecated.
        For rationale see:
        https://github.com/fonttools/fonttools/pull/2266/files#r613569473
        """
    vMetrics = getattr(ttFont.get('vmtx'), 'metrics', None)
    if vMetrics is None:
        verticalAdvanceWidth = ttFont['head'].unitsPerEm
        topSideY = getattr(ttFont.get('hhea'), 'ascent', None)
        if topSideY is None:
            if defaultVerticalOrigin is not None:
                topSideY = defaultVerticalOrigin
            else:
                topSideY = verticalAdvanceWidth
        glyph = self[glyphName]
        glyph.recalcBounds(self)
        topSideBearing = otRound(topSideY - glyph.yMax)
        vMetrics = {glyphName: (verticalAdvanceWidth, topSideBearing)}
    return vMetrics