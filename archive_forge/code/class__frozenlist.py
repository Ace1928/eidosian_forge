from __future__ import annotations
import functools
from .instruction import Instruction
from .gate import Gate
from .controlledgate import ControlledGate, _ctrl_state_to_int
class _frozenlist(list):
    __slots__ = ()

    def _reject_mutation(self, *args, **kwargs):
        raise TypeError("'params' of singletons cannot be mutated")
    append = clear = extend = insert = pop = remove = reverse = sort = _reject_mutation
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _reject_mutation