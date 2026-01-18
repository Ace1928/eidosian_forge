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
class ChainContextSubstBuilder(ChainContextualBuilder):
    """Builds a Chained Contextual Substitution (GSUB6) lookup.

    Users are expected to manually add rules to the ``rules`` attribute after
    the object has been initialized, e.g.::

        # sub [A B] [C D] x' lookup lu1 y' z' lookup lu2 E;

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
        LookupBuilder.__init__(self, font, location, 'GSUB', 6)
        self.rules = []
        self.subtable_type = 'Subst'

    def getAlternateGlyphs(self):
        result = {}
        for rule in self.rules:
            if rule.is_subtable_break:
                continue
            for lookups in rule.lookups:
                if not isinstance(lookups, list):
                    lookups = [lookups]
                for lookup in lookups:
                    if lookup is not None:
                        alts = lookup.getAlternateGlyphs()
                        for glyph, replacements in alts.items():
                            alts_for_glyph = result.setdefault(glyph, [])
                            alts_for_glyph.extend((g for g in replacements if g not in alts_for_glyph))
        return result

    def find_chainable_single_subst(self, mapping):
        """Helper for add_single_subst_chained_()"""
        res = None
        for rule in self.rules[::-1]:
            if rule.is_subtable_break:
                return res
            for sub in rule.lookups:
                if isinstance(sub, SingleSubstBuilder) and (not any((g in mapping and mapping[g] != sub.mapping[g] for g in sub.mapping))):
                    res = sub
        return res