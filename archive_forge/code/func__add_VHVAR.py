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
def _add_VHVAR(font, masterModel, master_ttfs, axisTags, tableFields):
    tableTag = tableFields.tableTag
    assert tableTag not in font
    log.info('Generating ' + tableTag)
    VHVAR = newTable(tableTag)
    tableClass = getattr(ot, tableTag)
    vhvar = VHVAR.table = tableClass()
    vhvar.Version = 65536
    glyphOrder = font.getGlyphOrder()
    metricsTag = tableFields.metricsTag
    advMetricses = [m[metricsTag].metrics for m in master_ttfs]
    if tableTag == 'VVAR' and 'VORG' in master_ttfs[0]:
        vOrigMetricses = [m['VORG'].VOriginRecords for m in master_ttfs]
        defaultYOrigs = [m['VORG'].defaultVertOriginY for m in master_ttfs]
        vOrigMetricses = list(zip(vOrigMetricses, defaultYOrigs))
    else:
        vOrigMetricses = None
    metricsStore, advanceMapping, vOrigMapping = _get_advance_metrics(font, masterModel, master_ttfs, axisTags, glyphOrder, advMetricses, vOrigMetricses)
    vhvar.VarStore = metricsStore
    if advanceMapping is None:
        setattr(vhvar, tableFields.advMapping, None)
    else:
        setattr(vhvar, tableFields.advMapping, advanceMapping)
    if vOrigMapping is not None:
        setattr(vhvar, tableFields.vOrigMapping, vOrigMapping)
    setattr(vhvar, tableFields.sb1, None)
    setattr(vhvar, tableFields.sb2, None)
    font[tableTag] = VHVAR
    return