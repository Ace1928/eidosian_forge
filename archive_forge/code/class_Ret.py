from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class Ret(Terminator):

    def __init__(self, parent, opname, return_value=None):
        operands = [return_value] if return_value is not None else []
        super(Ret, self).__init__(parent, opname, operands)

    @property
    def return_value(self):
        if self.operands:
            return self.operands[0]
        else:
            return None

    def descr(self, buf):
        return_value = self.return_value
        metadata = self._stringify_metadata(leading_comma=True)
        if return_value is not None:
            buf.append('{0} {1} {2}{3}\n'.format(self.opname, return_value.type, return_value.get_reference(), metadata))
        else:
            buf.append('{0}{1}\n'.format(self.opname, metadata))