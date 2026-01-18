from fontTools.misc.dictTools import hashdict
from fontTools.misc.intTools import bit_count
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools.otlLib.builder import buildLookup, buildSingleSubstSubtable
from collections import OrderedDict
from .errors import VarLibError, VarLibValidationError
def _existingVariableFeatures(table):
    existingFeatureVarsTags = set()
    if hasattr(table, 'FeatureVariations') and table.FeatureVariations is not None:
        features = table.FeatureList.FeatureRecord
        for fvr in table.FeatureVariations.FeatureVariationRecord:
            for ftsr in fvr.FeatureTableSubstitution.SubstitutionRecord:
                existingFeatureVarsTags.add(features[ftsr.FeatureIndex].FeatureTag)
    return existingFeatureVarsTags