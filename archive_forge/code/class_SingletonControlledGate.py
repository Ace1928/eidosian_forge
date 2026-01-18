from __future__ import annotations
import functools
from .instruction import Instruction
from .gate import Gate
from .controlledgate import ControlledGate, _ctrl_state_to_int
class SingletonControlledGate(ControlledGate, _SingletonBase, overrides=_SingletonControlledGateOverrides):
    """A base class to use for :class:`.ControlledGate` objects that by default are singleton instances

    This class is very similar to :class:`SingletonInstruction`, except implies unitary
    :class:`.ControlledGate` semantics as well.  The same caveats around setting attributes in
    that class apply here as well.
    """
    __slots__ = ()