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
def _decompileLigActions(self, actionReader, actionIndex):
    actions = []
    last = False
    reader = actionReader.getSubReader(actionReader.pos + actionIndex * 4)
    while not last:
        value = reader.readULong()
        last = bool(value & 2147483648)
        action = LigAction()
        actions.append(action)
        action.Store = bool(value & 1073741824)
        delta = value & 1073741823
        if delta >= 536870912:
            delta = -1073741824 + delta
        action.GlyphIndexDelta = delta
    return actions