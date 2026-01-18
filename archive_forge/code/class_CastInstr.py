from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class CastInstr(Instruction):

    def __init__(self, parent, op, val, typ, name=''):
        super(CastInstr, self).__init__(parent, typ, op, [val], name=name)

    def descr(self, buf):
        buf.append('{0} {1} {2} to {3} {4}\n'.format(self.opname, self.operands[0].type, self.operands[0].get_reference(), self.type, self._stringify_metadata(leading_comma=True)))