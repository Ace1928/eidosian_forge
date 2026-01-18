from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
def add_case(self, val, block):
    assert isinstance(block, Block)
    if not isinstance(val, Value):
        val = Constant(self.value.type, val)
    self.cases.append((val, block))