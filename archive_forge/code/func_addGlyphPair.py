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
def addGlyphPair(self, location, glyph1, value1, glyph2, value2):
    """Add a glyph pair positioning rule to the current lookup.

        Args:
            location: A string or tuple representing the location in the
                original source which produced this rule.
            glyph1: A glyph name for the "left" glyph in the pair.
            value1: A ``otTables.ValueRecord`` for positioning the left glyph.
            glyph2: A glyph name for the "right" glyph in the pair.
            value2: A ``otTables.ValueRecord`` for positioning the right glyph.
        """
    key = (glyph1, glyph2)
    oldValue = self.glyphPairs.get(key, None)
    if oldValue is not None:
        otherLoc = self.locations[key]
        log.debug('Already defined position for pair %s %s at %s; choosing the first value', glyph1, glyph2, otherLoc)
    else:
        self.glyphPairs[key] = (value1, value2)
        self.locations[key] = location