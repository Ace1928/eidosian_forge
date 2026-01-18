from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class CallInstrAttributes(AttributeSet):
    _known = frozenset(['convergent', 'noreturn', 'nounwind', 'readonly', 'readnone', 'noinline', 'alwaysinline'])