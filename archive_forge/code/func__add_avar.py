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
def _add_avar(font, axes, mappings, axisTags):
    """
    Add 'avar' table to font.

    axes is an ordered dictionary of AxisDescriptor objects.
    """
    assert axes
    assert isinstance(axes, OrderedDict)
    log.info('Generating avar')
    avar = newTable('avar')
    interesting = False
    vals_triples = {}
    for axis in axes.values():
        curve = avar.segments[axis.tag] = {-1.0: -1.0, 0.0: 0.0, 1.0: 1.0}
        keys_triple = (axis.minimum, axis.default, axis.maximum)
        vals_triple = tuple((axis.map_forward(v) for v in keys_triple))
        vals_triples[axis.tag] = vals_triple
        if not axis.map:
            continue
        items = sorted(axis.map)
        keys = [item[0] for item in items]
        vals = [item[1] for item in items]
        if axis.minimum != min(keys):
            raise VarLibValidationError(f"Axis '{axis.name}': there must be a mapping for the axis minimum value {axis.minimum} and it must be the lowest input mapping value.")
        if axis.maximum != max(keys):
            raise VarLibValidationError(f"Axis '{axis.name}': there must be a mapping for the axis maximum value {axis.maximum} and it must be the highest input mapping value.")
        if axis.default not in keys:
            raise VarLibValidationError(f"Axis '{axis.name}': there must be a mapping for the axis default value {axis.default}.")
        if len(set(keys)) != len(keys):
            raise VarLibValidationError(f"Axis '{axis.name}': All axis mapping input='...' values must be unique, but we found duplicates.")
        if sorted(vals) != vals:
            raise VarLibValidationError(f"Axis '{axis.name}': mapping output values must be in ascending order.")
        keys = [models.normalizeValue(v, keys_triple) for v in keys]
        vals = [models.normalizeValue(v, vals_triple) for v in vals]
        if all((k == v for k, v in zip(keys, vals))):
            continue
        interesting = True
        curve.update(zip(keys, vals))
        assert 0.0 in curve and curve[0.0] == 0.0
        assert -1.0 not in curve or curve[-1.0] == -1.0
        assert +1.0 not in curve or curve[+1.0] == +1.0
    if mappings:
        interesting = True
        inputLocations = [{axes[name].tag: models.normalizeValue(v, vals_triples[axes[name].tag]) for name, v in mapping.inputLocation.items()} for mapping in mappings]
        outputLocations = [{axes[name].tag: models.normalizeValue(v, vals_triples[axes[name].tag]) for name, v in mapping.outputLocation.items()} for mapping in mappings]
        assert len(inputLocations) == len(outputLocations)
        if not any((all((v == 0 for k, v in loc.items())) for loc in inputLocations)):
            inputLocations.insert(0, {})
            outputLocations.insert(0, {})
        model = models.VariationModel(inputLocations, axisTags)
        storeBuilder = varStore.OnlineVarStoreBuilder(axisTags)
        storeBuilder.setModel(model)
        varIdxes = {}
        for tag in axisTags:
            masterValues = []
            for vo, vi in zip(outputLocations, inputLocations):
                if tag not in vo:
                    masterValues.append(0)
                    continue
                v = vo[tag] - vi.get(tag, 0)
                masterValues.append(fl2fi(v, 14))
            varIdxes[tag] = storeBuilder.storeMasters(masterValues)[1]
        store = storeBuilder.finish()
        optimized = store.optimize()
        varIdxes = {axis: optimized[value] for axis, value in varIdxes.items()}
        varIdxMap = builder.buildDeltaSetIndexMap((varIdxes[t] for t in axisTags))
        avar.majorVersion = 2
        avar.table = ot.avar()
        avar.table.VarIdxMap = varIdxMap
        avar.table.VarStore = store
    assert 'avar' not in font
    if not interesting:
        log.info('No need for avar')
        avar = None
    else:
        font['avar'] = avar
    return avar