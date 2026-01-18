from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class SelectInstr(Instruction):

    def __init__(self, parent, cond, lhs, rhs, name='', flags=()):
        assert lhs.type == rhs.type
        super(SelectInstr, self).__init__(parent, lhs.type, 'select', [cond, lhs, rhs], name=name, flags=flags)

    @property
    def cond(self):
        return self.operands[0]

    @property
    def lhs(self):
        return self.operands[1]

    @property
    def rhs(self):
        return self.operands[2]

    def descr(self, buf):
        buf.append('select {0} {1} {2}, {3} {4}, {5} {6} {7}\n'.format(' '.join(self.flags), self.cond.type, self.cond.get_reference(), self.lhs.type, self.lhs.get_reference(), self.rhs.type, self.rhs.get_reference(), self._stringify_metadata(leading_comma=True)))