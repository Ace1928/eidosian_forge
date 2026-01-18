from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Mapping, Optional, Set, Tuple, cast
from warnings import warn
from pyquil.quil import Program
from pyquil.quilatom import ParameterDesignator, QubitDesignator, format_parameter
from pyquil.quilbase import (
def _build_generic_unitary(self) -> None:
    """
        Update the partial diagram with a unitary operation.

        Advances the index by one.
        """
    assert self.working_instructions is not None
    instr = self.working_instructions[self.index]
    assert isinstance(instr, Gate)
    qubits = qubit_indices(instr)
    dagger = sum((m == 'DAGGER' for m in instr.modifiers)) % 2 == 1
    controls = sum((m == 'CONTROLLED' for m in instr.modifiers))
    assert self.diagram is not None
    self.diagram.extend_lines_to_common_edge(qubits)
    control_qubits = qubits[:controls]
    target_qubits = sorted(qubits[controls:])
    if not self.diagram.is_interval(target_qubits):
        raise ValueError(f'Unable to render instruction {instr} which targets non-adjacent qubits.')
    for q in control_qubits:
        offset = target_qubits[0] - q
        self.diagram.append(q, TIKZ_CONTROL(q, offset))
    self.diagram.append(target_qubits[0], TIKZ_GATE(instr.name, size=len(target_qubits), params=instr.params, dagger=dagger))
    for q in target_qubits[1:]:
        self.diagram.append(q, TIKZ_NOP())
    self.index += 1