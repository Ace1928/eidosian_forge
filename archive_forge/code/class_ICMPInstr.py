from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class ICMPInstr(CompareInstr):
    OPNAME = 'icmp'
    VALID_OP = {'eq': 'equal', 'ne': 'not equal', 'ugt': 'unsigned greater than', 'uge': 'unsigned greater or equal', 'ult': 'unsigned less than', 'ule': 'unsigned less or equal', 'sgt': 'signed greater than', 'sge': 'signed greater or equal', 'slt': 'signed less than', 'sle': 'signed less or equal'}
    VALID_FLAG = set()