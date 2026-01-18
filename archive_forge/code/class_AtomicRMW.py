from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class AtomicRMW(Instruction):

    def __init__(self, parent, op, ptr, val, ordering, name):
        super(AtomicRMW, self).__init__(parent, val.type, 'atomicrmw', (ptr, val), name=name)
        self.operation = op
        self.ordering = ordering

    def descr(self, buf):
        ptr, val = self.operands
        fmt = 'atomicrmw {op} {ptrty} {ptr}, {valty} {val} {ordering} {metadata}\n'
        buf.append(fmt.format(op=self.operation, ptrty=ptr.type, ptr=ptr.get_reference(), valty=val.type, val=val.get_reference(), ordering=self.ordering, metadata=self._stringify_metadata(leading_comma=True)))