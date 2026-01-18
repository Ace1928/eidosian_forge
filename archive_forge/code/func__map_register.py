from __future__ import annotations
import typing
from .bit import Bit
from .classical import expr
from .classicalregister import ClassicalRegister, Clbit
def _map_register(self, theirs: ClassicalRegister) -> ClassicalRegister:
    """Map the target's registers to suitable equivalents in the destination, adding an
        extra one if there's no exact match."""
    if (mapped_theirs := self.register_map.get(theirs.name)) is not None:
        return mapped_theirs
    mapped_bits = [self.bit_map[bit] for bit in theirs]
    for ours in self.target_cregs:
        if mapped_bits == list(ours):
            mapped_theirs = ours
            break
    else:
        if self.add_register is None:
            raise ValueError(f"Register '{theirs.name}' has no counterpart in the destination.")
        mapped_theirs = ClassicalRegister(bits=mapped_bits)
        self.add_register(mapped_theirs)
    self.register_map[theirs.name] = mapped_theirs
    return mapped_theirs