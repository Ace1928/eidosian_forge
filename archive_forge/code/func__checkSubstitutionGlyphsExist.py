from fontTools.misc.dictTools import hashdict
from fontTools.misc.intTools import bit_count
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools.otlLib.builder import buildLookup, buildSingleSubstSubtable
from collections import OrderedDict
from .errors import VarLibError, VarLibValidationError
def _checkSubstitutionGlyphsExist(glyphNames, substitutions):
    referencedGlyphNames = set()
    for _, substitution in substitutions:
        referencedGlyphNames |= substitution.keys()
        referencedGlyphNames |= set(substitution.values())
    missing = referencedGlyphNames - glyphNames
    if missing:
        raise VarLibValidationError(f'Missing glyphs are referenced in conditional substitution rules: {', '.join(missing)}')