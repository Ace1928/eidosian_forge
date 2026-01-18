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
class ChainContextPosBuilder(ChainContextualBuilder):
    """Builds a Chained Contextual Positioning (GPOS8) lookup.

    Users are expected to manually add rules to the ``rules`` attribute after
    the object has been initialized, e.g.::

        # pos [A B] [C D] x' lookup lu1 y' z' lookup lu2 E;

        prefix  = [ ["A", "B"], ["C", "D"] ]
        suffix  = [ ["E"] ]
        glyphs  = [ ["x"], ["y"], ["z"] ]
        lookups = [ [lu1], None,  [lu2] ]
        builder.rules.append( (prefix, glyphs, suffix, lookups) )

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        rules: A list of tuples representing the rules in this lookup.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, 'GPOS', 8)
        self.rules = []
        self.subtable_type = 'Pos'

    def find_chainable_single_pos(self, lookups, glyphs, value):
        """Helper for add_single_pos_chained_()"""
        res = None
        for lookup in lookups[::-1]:
            if lookup == self.SUBTABLE_BREAK_:
                return res
            if isinstance(lookup, SinglePosBuilder) and all((lookup.can_add(glyph, value) for glyph in glyphs)):
                res = lookup
        return res