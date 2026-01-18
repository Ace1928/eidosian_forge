import collections
import copy
import enum
from functools import partial
from math import ceil, log
from typing import (
from fontTools.misc.arrayTools import intRect
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.treeTools import build_n_ary_tree
from fontTools.ttLib.tables import C_O_L_R_
from fontTools.ttLib.tables import C_P_A_L_
from fontTools.ttLib.tables import _n_a_m_e
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otTables import ExtendMode, CompositeMode
from .errors import ColorLibError
from .geometry import round_start_circle_stable_containment
from .table_builder import BuildCallback, TableBuilder
def _beforeBuildPaintRadialGradient(paint, source):
    x0 = source['x0']
    y0 = source['y0']
    r0 = source['r0']
    x1 = source['x1']
    y1 = source['y1']
    r1 = source['r1']
    c = round_start_circle_stable_containment((x0, y0), r0, (x1, y1), r1)
    x0, y0 = c.centre
    r0 = c.radius
    source['x0'] = x0
    source['y0'] = y0
    source['r0'] = r0
    source['x1'] = x1
    source['y1'] = y1
    source['r1'] = r1
    return (paint, source)