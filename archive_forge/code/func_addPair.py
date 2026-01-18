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
def addPair(self, gc1, value1, gc2, value2):
    """Add a pair positioning rule.

        Args:
            gc1: A set of glyph names for the "left" glyph
            value1: An ``otTables.ValueRecord`` object for the left glyph's
                positioning.
            gc2: A set of glyph names for the "right" glyph
            value2: An ``otTables.ValueRecord`` object for the right glyph's
                positioning.
        """
    mergeable = not self.forceSubtableBreak_ and self.classDef1_ is not None and self.classDef1_.canAdd(gc1) and (self.classDef2_ is not None) and self.classDef2_.canAdd(gc2)
    if not mergeable:
        self.flush_()
        self.classDef1_ = ClassDefBuilder(useClass0=True)
        self.classDef2_ = ClassDefBuilder(useClass0=False)
        self.values_ = {}
    self.classDef1_.add(gc1)
    self.classDef2_.add(gc2)
    self.values_[gc1, gc2] = (value1, value2)