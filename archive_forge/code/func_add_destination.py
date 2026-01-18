from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
def add_destination(self, block):
    assert isinstance(block, Block)
    self.destinations.append(block)