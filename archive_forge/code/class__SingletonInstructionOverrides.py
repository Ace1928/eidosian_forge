from __future__ import annotations
import functools
from .instruction import Instruction
from .gate import Gate
from .controlledgate import ControlledGate, _ctrl_state_to_int
class _SingletonInstructionOverrides(Instruction):
    """Overrides for the mutable methods and properties of `Instruction` to make it immutable."""
    __slots__ = ()

    @staticmethod
    def _prepare_singleton_instance(instruction: Instruction):
        """Class-creation hook point.  Given an instance of the type that these overrides correspond
        to, this method should ensure that all lazy properties and caches that require mutation to
        write to are eagerly defined.

        Subclass "overrides" classes can override this method if the user/library-author-facing
        class they are providing overrides for has more lazy attributes or user-exposed state
        with interior mutability."""
        instruction._define()
        instruction._params = _frozenlist(instruction._params)
        return instruction

    def c_if(self, classical, val):
        return self.to_mutable().c_if(classical, val)

    def copy(self, name=None):
        if name is None:
            return self
        out = self.to_mutable()
        out.name = name
        return out