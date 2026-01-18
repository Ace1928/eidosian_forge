import logging
import re
from dataclasses import dataclass
from typing import Any, FrozenSet, Generator, Iterable, List, Optional, cast
from pyquil.paulis import PauliTerm, sI
def _in_operator(self) -> PauliTerm:
    pt = sI()
    for oneq_state in self.in_state.states:
        if oneq_state.label not in ['X', 'Y', 'Z']:
            raise ValueError(f"Can't shim {oneq_state.label} into a pauli term. Use in_state.")
        if oneq_state.index != 0:
            raise ValueError(f"Can't shim {oneq_state} into a pauli term. Use in_state.")
        new_pt = pt * PauliTerm(op=oneq_state.label, index=oneq_state.qubit)
        pt = cast(PauliTerm, new_pt)
    return pt