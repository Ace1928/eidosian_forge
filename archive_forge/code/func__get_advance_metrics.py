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
def _get_advance_metrics(font, masterModel, master_ttfs, axisTags, glyphOrder, advMetricses, vOrigMetricses=None):
    vhAdvanceDeltasAndSupports = {}
    vOrigDeltasAndSupports = {}
    sparse_advance = 65535
    for glyph in glyphOrder:
        vhAdvances = [metrics[glyph][0] if glyph in metrics and metrics[glyph][0] != sparse_advance else None for metrics in advMetricses]
        vhAdvanceDeltasAndSupports[glyph] = masterModel.getDeltasAndSupports(vhAdvances, round=round)
    singleModel = models.allEqual((id(v[1]) for v in vhAdvanceDeltasAndSupports.values()))
    if vOrigMetricses:
        singleModel = False
        for glyph in glyphOrder:
            vOrigs = [metrics[glyph] if glyph in metrics else defaultVOrig for metrics, defaultVOrig in vOrigMetricses]
            vOrigDeltasAndSupports[glyph] = masterModel.getDeltasAndSupports(vOrigs, round=round)
    directStore = None
    if singleModel:
        supports = next(iter(vhAdvanceDeltasAndSupports.values()))[1][1:]
        varTupleList = builder.buildVarRegionList(supports, axisTags)
        varTupleIndexes = list(range(len(supports)))
        varData = builder.buildVarData(varTupleIndexes, [], optimize=False)
        for glyphName in glyphOrder:
            varData.addItem(vhAdvanceDeltasAndSupports[glyphName][0], round=noRound)
        varData.optimize()
        directStore = builder.buildVarStore(varTupleList, [varData])
    storeBuilder = varStore.OnlineVarStoreBuilder(axisTags)
    advMapping = {}
    for glyphName in glyphOrder:
        deltas, supports = vhAdvanceDeltasAndSupports[glyphName]
        storeBuilder.setSupports(supports)
        advMapping[glyphName] = storeBuilder.storeDeltas(deltas, round=noRound)
    if vOrigMetricses:
        vOrigMap = {}
        for glyphName in glyphOrder:
            deltas, supports = vOrigDeltasAndSupports[glyphName]
            storeBuilder.setSupports(supports)
            vOrigMap[glyphName] = storeBuilder.storeDeltas(deltas, round=noRound)
    indirectStore = storeBuilder.finish()
    mapping2 = indirectStore.optimize(use_NO_VARIATION_INDEX=False)
    advMapping = [mapping2[advMapping[g]] for g in glyphOrder]
    advanceMapping = builder.buildVarIdxMap(advMapping, glyphOrder)
    if vOrigMetricses:
        vOrigMap = [mapping2[vOrigMap[g]] for g in glyphOrder]
    useDirect = False
    vOrigMapping = None
    if directStore:
        writer = OTTableWriter()
        directStore.compile(writer, font)
        directSize = len(writer.getAllData())
        writer = OTTableWriter()
        indirectStore.compile(writer, font)
        advanceMapping.compile(writer, font)
        indirectSize = len(writer.getAllData())
        useDirect = directSize < indirectSize
    if useDirect:
        metricsStore = directStore
        advanceMapping = None
    else:
        metricsStore = indirectStore
        if vOrigMetricses:
            vOrigMapping = builder.buildVarIdxMap(vOrigMap, glyphOrder)
    return (metricsStore, advanceMapping, vOrigMapping)