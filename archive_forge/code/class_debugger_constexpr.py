from __future__ import annotations
from ..language import core as lcore
from . import torch_wrapper
from .core import ExecutionContext
from .memory_map import MemoryMap
class debugger_constexpr:

    def __init__(self, value):
        if isinstance(value, debugger_constexpr):
            self.value = value.value
        else:
            self.value = value

    def __str__(self) -> str:
        return 'debugger_constexpr(' + str(self.value) + ')'

    def __index__(self) -> int:
        return self.value

    def __bool__(self):
        return bool(self.value)

    def __ge__(self, other):
        other = other.value if isinstance(other, debugger_constexpr) else other
        return self.value >= other

    def __gt__(self, other):
        other = other.value if isinstance(other, debugger_constexpr) else other
        return self.value > other

    def __le__(self, other):
        other = other.value if isinstance(other, debugger_constexpr) else other
        return self.value <= other

    def __lt__(self, other):
        other = other.value if isinstance(other, debugger_constexpr) else other
        return self.value < other

    def __eq__(self, other):
        other = other.value if isinstance(other, debugger_constexpr) else other
        return self.value == other

    def __or__(self, other):
        other = other.value if isinstance(other, debugger_constexpr) else other
        return self.value | other

    def __ror__(self, other):
        other = other.value if isinstance(other, debugger_constexpr) else other
        return self.value | other

    def __and__(self, other):
        other = other.value if isinstance(other, debugger_constexpr) else other
        return self.value & other

    def __rand__(self, other):
        other = other.value if isinstance(other, debugger_constexpr) else other
        return self.value & other

    def to(self, dtype, bitcast=False, _builder=None):
        if dtype in [torch.int64]:
            ret_ty = int
        elif dtype == torch.bool:
            ret_ty = bool
        elif dtype in [torch.float64]:
            ret_ty = float
        else:
            raise ValueError('dtype not supported in debugger')
        return debugger_constexpr(ret_ty(self.value))