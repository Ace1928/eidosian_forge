from typing import List
from fontTools.misc.vector import Vector
from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.fixedTools import floatToFixed as fl2fi
from fontTools.misc.textTools import Tag, tostr
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables._f_v_a_r import Axis, NamedInstance
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates, dropImpliedOnCurvePoints
from fontTools.ttLib.tables.ttProgram import Program
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.varLib import builder, models, varStore
from fontTools.varLib.merger import VariationMerger, COLRVariationMerger
from fontTools.varLib.mvar import MVAR_ENTRIES
from fontTools.varLib.iup import iup_delta_optimize
from fontTools.varLib.featureVars import addFeatureVariations
from fontTools.designspaceLib import DesignSpaceDocument, InstanceDescriptor
from fontTools.designspaceLib.split import splitInterpolable, splitVariableFonts
from fontTools.varLib.stat import buildVFStatTable
from fontTools.colorLib.builder import buildColrV1
from fontTools.colorLib.unbuilder import unbuildColrV1
from functools import partial
from collections import OrderedDict, defaultdict, namedtuple
import os.path
import logging
from copy import deepcopy
from pprint import pformat
from re import fullmatch
from .errors import VarLibError, VarLibValidationError
def addGSUBFeatureVariations(vf, designspace, featureTags=(), *, log_enabled=False):
    """Add GSUB FeatureVariations table to variable font, based on DesignSpace rules.

    Args:
        vf: A TTFont object representing the variable font.
        designspace: A DesignSpaceDocument object.
        featureTags: Optional feature tag(s) to use for the FeatureVariations records.
            If unset, the key 'com.github.fonttools.varLib.featureVarsFeatureTag' is
            looked up in the DS <lib> and used; otherwise the default is 'rclt' if
            the <rules processing="last"> attribute is set, else 'rvrn'.
            See <https://fonttools.readthedocs.io/en/latest/designspaceLib/xml.html#rules-element>
        log_enabled: If True, log info about DS axes and sources. Default is False, as
            the same info may have already been logged as part of varLib.build.
    """
    ds = load_designspace(designspace, log_enabled=log_enabled)
    if not ds.rules:
        return
    if not featureTags:
        featureTags = _feature_variations_tags(ds)
    _add_GSUB_feature_variations(vf, ds.axes, ds.internal_axis_supports, ds.rules, featureTags)