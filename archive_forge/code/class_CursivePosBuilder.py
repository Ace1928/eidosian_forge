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
class CursivePosBuilder(LookupBuilder):
    """Builds a Cursive Positioning (GPOS3) lookup.

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        attachments: An ordered dictionary mapping a glyph name to a two-element
            tuple of ``otTables.Anchor`` objects.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, 'GPOS', 3)
        self.attachments = {}

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.attachments == other.attachments

    def add_attachment(self, location, glyphs, entryAnchor, exitAnchor):
        """Adds attachment information to the cursive positioning lookup.

        Args:
            location: A string or tuple representing the location in the
                original source which produced this lookup. (Unused.)
            glyphs: A list of glyph names sharing these entry and exit
                anchor locations.
            entryAnchor: A ``otTables.Anchor`` object representing the
                entry anchor, or ``None`` if no entry anchor is present.
            exitAnchor: A ``otTables.Anchor`` object representing the
                exit anchor, or ``None`` if no exit anchor is present.
        """
        for glyph in glyphs:
            self.attachments[glyph] = (entryAnchor, exitAnchor)

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the cursive
            positioning lookup.
        """
        st = buildCursivePosSubtable(self.attachments, self.glyphMap)
        return self.buildLookup_([st])