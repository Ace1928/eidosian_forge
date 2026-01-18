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
def _getPhantomPoints(self, glyphName, hMetrics, vMetrics=None):
    """Compute the four "phantom points" for the given glyph from its bounding box
        and the horizontal and vertical advance widths and sidebearings stored in the
        ttFont's "hmtx" and "vmtx" tables.

        'hMetrics' should be ttFont['hmtx'].metrics.

        'vMetrics' should be ttFont['vmtx'].metrics if there is "vmtx" or None otherwise.
        If there is no vMetrics passed in, vertical phantom points are set to the zero coordinate.

        https://docs.microsoft.com/en-us/typography/opentype/spec/tt_instructing_glyphs#phantoms
        """
    glyph = self[glyphName]
    if not hasattr(glyph, 'xMin'):
        glyph.recalcBounds(self)
    horizontalAdvanceWidth, leftSideBearing = hMetrics[glyphName]
    leftSideX = glyph.xMin - leftSideBearing
    rightSideX = leftSideX + horizontalAdvanceWidth
    if vMetrics:
        verticalAdvanceWidth, topSideBearing = vMetrics[glyphName]
        topSideY = topSideBearing + glyph.yMax
        bottomSideY = topSideY - verticalAdvanceWidth
    else:
        bottomSideY = topSideY = 0
    return [(leftSideX, 0), (rightSideX, 0), (0, topSideY), (0, bottomSideY)]