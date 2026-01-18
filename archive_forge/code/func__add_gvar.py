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
def _add_gvar(font, masterModel, master_ttfs, tolerance=0.5, optimize=True):
    if tolerance < 0:
        raise ValueError('`tolerance` must be a positive number.')
    log.info('Generating gvar')
    assert 'gvar' not in font
    gvar = font['gvar'] = newTable('gvar')
    glyf = font['glyf']
    defaultMasterIndex = masterModel.reverseMapping[0]
    master_datas = [_MasterData(m['glyf'], m['hmtx'].metrics, getattr(m.get('vmtx'), 'metrics', None)) for m in master_ttfs]
    for glyph in font.getGlyphOrder():
        log.debug("building gvar for glyph '%s'", glyph)
        isComposite = glyf[glyph].isComposite()
        allData = [m.glyf._getCoordinatesAndControls(glyph, m.hMetrics, m.vMetrics) for m in master_datas]
        if allData[defaultMasterIndex][1].numberOfContours != 0:
            allData = [d if d is not None and d[1].numberOfContours != 0 else None for d in allData]
        model, allData = masterModel.getSubModel(allData)
        allCoords = [d[0] for d in allData]
        allControls = [d[1] for d in allData]
        control = allControls[0]
        if not models.allEqual(allControls):
            log.warning('glyph %s has incompatible masters; skipping' % glyph)
            continue
        del allControls
        gvar.variations[glyph] = []
        deltas = model.getDeltas(allCoords, round=partial(GlyphCoordinates.__round__, round=round))
        supports = model.supports
        assert len(deltas) == len(supports)
        origCoords = deltas[0]
        endPts = control.endPts
        for i, (delta, support) in enumerate(zip(deltas[1:], supports[1:])):
            if all((v == 0 for v in delta.array)) and (not isComposite):
                continue
            var = TupleVariation(support, delta)
            if optimize:
                delta_opt = iup_delta_optimize(delta, origCoords, endPts, tolerance=tolerance)
                if None in delta_opt:
                    'In composite glyphs, there should be one 0 entry\n                    to make sure the gvar entry is written to the font.\n\n                    This is to work around an issue with macOS 10.14 and can be\n                    removed once the behaviour of macOS is changed.\n\n                    https://github.com/fonttools/fonttools/issues/1381\n                    '
                    if all((d is None for d in delta_opt)):
                        delta_opt = [(0, 0)] + [None] * (len(delta_opt) - 1)
                    var_opt = TupleVariation(support, delta_opt)
                    axis_tags = sorted(support.keys())
                    tupleData, auxData = var.compile(axis_tags)
                    unoptimized_len = len(tupleData) + len(auxData)
                    tupleData, auxData = var_opt.compile(axis_tags)
                    optimized_len = len(tupleData) + len(auxData)
                    if optimized_len < unoptimized_len:
                        var = var_opt
            gvar.variations[glyph].append(var)