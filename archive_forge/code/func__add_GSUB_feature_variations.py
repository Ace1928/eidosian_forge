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
def _add_GSUB_feature_variations(font, axes, internal_axis_supports, rules, featureTags):

    def normalize(name, value):
        return models.normalizeLocation({name: value}, internal_axis_supports)[name]
    log.info('Generating GSUB FeatureVariations')
    axis_tags = {name: axis.tag for name, axis in axes.items()}
    conditional_subs = []
    for rule in rules:
        region = []
        for conditions in rule.conditionSets:
            space = {}
            for condition in conditions:
                axis_name = condition['name']
                if condition['minimum'] is not None:
                    minimum = normalize(axis_name, condition['minimum'])
                else:
                    minimum = -1.0
                if condition['maximum'] is not None:
                    maximum = normalize(axis_name, condition['maximum'])
                else:
                    maximum = 1.0
                tag = axis_tags[axis_name]
                space[tag] = (minimum, maximum)
            region.append(space)
        subs = {k: v for k, v in rule.subs}
        conditional_subs.append((region, subs))
    addFeatureVariations(font, conditional_subs, featureTags)