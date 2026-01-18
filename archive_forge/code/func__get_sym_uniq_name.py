import math
import ctypes
import copy
from .random import uniform
from .symbol import Symbol
from . import symbol
from ..base import _LIB, check_call
from ..base import SymbolHandle, _as_list
from ..attribute import AttrScope
def _get_sym_uniq_name(sym):
    return '{}-{}'.format(sym.name, sym.attr('_value_index'))