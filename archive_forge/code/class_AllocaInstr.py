from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class AllocaInstr(Instruction):

    def __init__(self, parent, typ, count, name):
        operands = [count] if count else ()
        super(AllocaInstr, self).__init__(parent, typ.as_pointer(), 'alloca', operands, name)
        self.align = None

    def descr(self, buf):
        buf.append('{0} {1}'.format(self.opname, self.type.pointee))
        if self.operands:
            op, = self.operands
            buf.append(', {0} {1}'.format(op.type, op.get_reference()))
        if self.align is not None:
            buf.append(', align {0}'.format(self.align))
        if self.metadata:
            buf.append(self._stringify_metadata(leading_comma=True))