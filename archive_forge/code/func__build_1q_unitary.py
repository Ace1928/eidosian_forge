from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Mapping, Optional, Set, Tuple, cast
from warnings import warn
from pyquil.quil import Program
from pyquil.quilatom import ParameterDesignator, QubitDesignator, format_parameter
from pyquil.quilbase import (
def _build_1q_unitary(self) -> None:
    """
        Update the partial diagram with a 1Q gate.

        Advances the index by one.
        """
    assert self.working_instructions is not None
    instr = self.working_instructions[self.index]
    assert isinstance(instr, Gate)
    qubits = qubit_indices(instr)
    dagger = sum((m == 'DAGGER' for m in instr.modifiers)) % 2 == 1
    assert self.diagram is not None
    self.diagram.append(qubits[0], TIKZ_GATE(instr.name, params=instr.params, dagger=dagger, settings=self.settings))
    self.index += 1