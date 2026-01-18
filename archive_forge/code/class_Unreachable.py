from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class Unreachable(Instruction):

    def __init__(self, parent):
        super(Unreachable, self).__init__(parent, types.VoidType(), 'unreachable', (), name='')

    def descr(self, buf):
        buf += (self.opname, '\n')