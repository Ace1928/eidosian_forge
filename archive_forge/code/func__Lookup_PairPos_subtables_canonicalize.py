import os
import copy
import enum
from operator import ior
import logging
from fontTools.colorLib.builder import MAX_PAINT_COLR_LAYER_COUNT, LayerReuseCache
from fontTools.misc import classifyTools
from fontTools.misc.roundTools import otRound
from fontTools.misc.treeTools import build_n_ary_tree
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables import otBase as otBase
from fontTools.ttLib.tables.otConverters import BaseFixedValue
from fontTools.ttLib.tables.otTraverse import dfs_base_table
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.varLib import builder, models, varStore
from fontTools.varLib.models import nonNone, allNone, allEqual, allEqualTo, subList
from fontTools.varLib.varStore import VarStoreInstancer
from functools import reduce
from fontTools.otlLib.builder import buildSinglePos
from fontTools.otlLib.optimize.gpos import (
from .errors import (
def _Lookup_PairPos_subtables_canonicalize(lst, font):
    """Merge multiple Format1 subtables at the beginning of lst,
    and merge multiple consecutive Format2 subtables that have the same
    Class2 (ie. were split because of offset overflows).  Returns new list."""
    lst = list(lst)
    l = len(lst)
    i = 0
    while i < l and lst[i].Format == 1:
        i += 1
    lst[:i] = [_Lookup_PairPosFormat1_subtables_flatten(lst[:i], font)]
    l = len(lst)
    i = l
    while i > 0 and lst[i - 1].Format == 2:
        i -= 1
    lst[i:] = [_Lookup_PairPosFormat2_subtables_flatten(lst[i:], font)]
    return lst