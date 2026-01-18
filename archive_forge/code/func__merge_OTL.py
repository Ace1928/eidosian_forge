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
def _merge_OTL(font, model, master_fonts, axisTags):
    otl_tags = ['GSUB', 'GDEF', 'GPOS']
    if not any((tag in font for tag in otl_tags)):
        return
    log.info('Merging OpenType Layout tables')
    merger = VariationMerger(model, axisTags, font)
    merger.mergeTables(font, master_fonts, otl_tags)
    store = merger.store_builder.finish()
    if not store:
        return
    try:
        GDEF = font['GDEF'].table
        assert GDEF.Version <= 65538
    except KeyError:
        font['GDEF'] = newTable('GDEF')
        GDEFTable = font['GDEF'] = newTable('GDEF')
        GDEF = GDEFTable.table = ot.GDEF()
        GDEF.GlyphClassDef = None
        GDEF.AttachList = None
        GDEF.LigCaretList = None
        GDEF.MarkAttachClassDef = None
        GDEF.MarkGlyphSetsDef = None
    GDEF.Version = 65539
    GDEF.VarStore = store
    varidx_map = store.optimize()
    GDEF.remap_device_varidxes(varidx_map)
    if 'GPOS' in font:
        font['GPOS'].table.remap_device_varidxes(varidx_map)