from fontTools.misc.dictTools import hashdict
from fontTools.misc.intTools import bit_count
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools.otlLib.builder import buildLookup, buildSingleSubstSubtable
from collections import OrderedDict
from .errors import VarLibError, VarLibValidationError
def addFeatureVariations(font, conditionalSubstitutions, featureTag='rvrn'):
    """Add conditional substitutions to a Variable Font.

    The `conditionalSubstitutions` argument is a list of (Region, Substitutions)
    tuples.

    A Region is a list of Boxes. A Box is a dict mapping axisTags to
    (minValue, maxValue) tuples. Irrelevant axes may be omitted and they are
    interpretted as extending to end of axis in each direction.  A Box represents
    an orthogonal 'rectangular' subset of an N-dimensional design space.
    A Region represents a more complex subset of an N-dimensional design space,
    ie. the union of all the Boxes in the Region.
    For efficiency, Boxes within a Region should ideally not overlap, but
    functionality is not compromised if they do.

    The minimum and maximum values are expressed in normalized coordinates.

    A Substitution is a dict mapping source glyph names to substitute glyph names.

    Example:

    # >>> f = TTFont(srcPath)
    # >>> condSubst = [
    # ...     # A list of (Region, Substitution) tuples.
    # ...     ([{"wdth": (0.5, 1.0)}], {"cent": "cent.rvrn"}),
    # ...     ([{"wght": (0.5, 1.0)}], {"dollar": "dollar.rvrn"}),
    # ... ]
    # >>> addFeatureVariations(f, condSubst)
    # >>> f.save(dstPath)

    The `featureTag` parameter takes either a str or a iterable of str (the single str
    is kept for backwards compatibility), and defines which feature(s) will be
    associated with the feature variations.
    Note, if this is "rvrn", then the substitution lookup will be inserted at the
    beginning of the lookup list so that it is processed before others, otherwise
    for any other feature tags it will be appended last.
    """
    featureTags = [featureTag] if isinstance(featureTag, str) else sorted(featureTag)
    processLast = 'rvrn' not in featureTags or len(featureTags) > 1
    _checkSubstitutionGlyphsExist(glyphNames=set(font.getGlyphOrder()), substitutions=conditionalSubstitutions)
    substitutions = overlayFeatureVariations(conditionalSubstitutions)
    conditionalSubstitutions, allSubstitutions = makeSubstitutionsHashable(substitutions)
    if 'GSUB' not in font:
        font['GSUB'] = buildGSUB()
    else:
        existingTags = _existingVariableFeatures(font['GSUB'].table).intersection(featureTags)
        if existingTags:
            raise VarLibError(f'FeatureVariations already exist for feature tag(s): {existingTags}')
    lookupMap = buildSubstitutionLookups(font['GSUB'].table, allSubstitutions, processLast)
    conditionsAndLookups = []
    for conditionSet, substitutions in conditionalSubstitutions:
        conditionsAndLookups.append((conditionSet, [lookupMap[s] for s in substitutions]))
    addFeatureVariationsRaw(font, font['GSUB'].table, conditionsAndLookups, featureTags)