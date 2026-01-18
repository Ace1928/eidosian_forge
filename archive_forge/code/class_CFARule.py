import copy
from collections import namedtuple
from ..common.utils import (
from ..construct import Struct, Switch
from .enums import DW_EH_encoding_flags
from .structs import DWARFStructs
from .constants import *
class CFARule(object):
    """ A CFA rule is used to compute the CFA for each location. It either
        consists of a register+offset, or a DWARF expression.
    """

    def __init__(self, reg=None, offset=None, expr=None):
        self.reg = reg
        self.offset = offset
        self.expr = expr

    def __repr__(self):
        return 'CFARule(reg=%s, offset=%s, expr=%s)' % (self.reg, self.offset, self.expr)