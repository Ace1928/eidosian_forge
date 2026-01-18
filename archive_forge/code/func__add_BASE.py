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
def _add_BASE(font, masterModel, master_ttfs, axisTags):
    log.info('Generating BASE')
    merger = VariationMerger(masterModel, axisTags, font)
    merger.mergeTables(font, master_ttfs, ['BASE'])
    store = merger.store_builder.finish()
    if not store:
        return
    base = font['BASE'].table
    assert base.Version == 65536
    base.Version = 65537
    base.VarStore = store