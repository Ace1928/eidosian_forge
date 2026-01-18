from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class StoreInstr(Instruction):

    def __init__(self, parent, val, ptr):
        super(StoreInstr, self).__init__(parent, types.VoidType(), 'store', [val, ptr])

    def descr(self, buf):
        val, ptr = self.operands
        if self.align is not None:
            align = ', align %d' % self.align
        else:
            align = ''
        buf.append('store {0} {1}, {2} {3}{4}{5}\n'.format(val.type, val.get_reference(), ptr.type, ptr.get_reference(), align, self._stringify_metadata(leading_comma=True)))