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
def _tuple_safe(value):
    if isinstance(value, enum.Enum):
        return value
    elif hasattr(value, '__dict__'):
        return tuple(((k, _tuple_safe(v)) for k, v in sorted(value.__dict__.items())))
    elif isinstance(value, collections.abc.MutableSequence):
        return tuple((_tuple_safe(e) for e in value))
    return value