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
class PaintFormat(IntEnum):
    PaintColrLayers = 1
    PaintSolid = 2
    PaintVarSolid = 3
    PaintLinearGradient = 4
    PaintVarLinearGradient = 5
    PaintRadialGradient = 6
    PaintVarRadialGradient = 7
    PaintSweepGradient = 8
    PaintVarSweepGradient = 9
    PaintGlyph = 10
    PaintColrGlyph = 11
    PaintTransform = 12
    PaintVarTransform = 13
    PaintTranslate = 14
    PaintVarTranslate = 15
    PaintScale = 16
    PaintVarScale = 17
    PaintScaleAroundCenter = 18
    PaintVarScaleAroundCenter = 19
    PaintScaleUniform = 20
    PaintVarScaleUniform = 21
    PaintScaleUniformAroundCenter = 22
    PaintVarScaleUniformAroundCenter = 23
    PaintRotate = 24
    PaintVarRotate = 25
    PaintRotateAroundCenter = 26
    PaintVarRotateAroundCenter = 27
    PaintSkew = 28
    PaintVarSkew = 29
    PaintSkewAroundCenter = 30
    PaintVarSkewAroundCenter = 31
    PaintComposite = 32

    def is_variable(self):
        return self.name.startswith('PaintVar')

    def as_variable(self):
        if self.is_variable():
            return self
        try:
            return PaintFormat.__members__[f'PaintVar{self.name[5:]}']
        except KeyError:
            return None