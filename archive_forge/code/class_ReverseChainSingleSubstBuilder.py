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
class ReverseChainSingleSubstBuilder(LookupBuilder):
    """Builds a Reverse Chaining Contextual Single Substitution (GSUB8) lookup.

    Users are expected to manually add substitutions to the ``substitutions``
    attribute after the object has been initialized, e.g.::

        # reversesub [a e n] d' by d.alt;
        prefix = [ ["a", "e", "n"] ]
        suffix = []
        mapping = { "d": "d.alt" }
        builder.substitutions.append( (prefix, suffix, mapping) )

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        substitutions: A three-element tuple consisting of a prefix sequence,
            a suffix sequence, and a dictionary of single substitutions.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, 'GSUB', 8)
        self.rules = []

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.rules == other.rules

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the chained
            contextual substitution lookup.
        """
        subtables = []
        for prefix, suffix, mapping in self.rules:
            st = ot.ReverseChainSingleSubst()
            st.Format = 1
            self.setBacktrackCoverage_(prefix, st)
            self.setLookAheadCoverage_(suffix, st)
            st.Coverage = buildCoverage(mapping.keys(), self.glyphMap)
            st.GlyphCount = len(mapping)
            st.Substitute = [mapping[g] for g in st.Coverage.glyphs]
            subtables.append(st)
        return self.buildLookup_(subtables)

    def add_subtable_break(self, location):
        pass