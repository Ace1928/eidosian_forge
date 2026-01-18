from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable, List, Sequence, Mapping, Optional, Set, Tuple, cast
from warnings import warn
from pyquil.quil import Program
from pyquil.quilatom import ParameterDesignator, QubitDesignator, format_parameter
from pyquil.quilbase import (
class DiagramBuilder:
    """
    Constructs DiagramStates from a given circuit and settings.

    This is essentially a state machine, represented by a few instance variables and some mutually
    recursive methods.
    """

    def __init__(self, circuit: Program, settings: DiagramSettings):
        self.circuit = circuit
        self.settings = settings
        self.working_instructions: Optional[List[AbstractInstruction]] = None
        self.index = 0
        self.diagram: Optional[DiagramState] = None

    def build(self) -> DiagramState:
        """
        Actually build the diagram.
        """
        qubits = cast(Set[int], self.circuit.get_qubits(indices=True))
        all_qubits = range(min(qubits), max(qubits) + 1) if self.settings.impute_missing_qubits else sorted(qubits)
        self.diagram = DiagramState(all_qubits)
        if self.settings.right_align_terminal_measurements:
            measures, instructions = split_on_terminal_measures(self.circuit)
        else:
            measures, instructions = ([], self.circuit.instructions)
        if self.settings.label_qubit_lines:
            for qubit in self.diagram.qubits:
                self.diagram.append(qubit, TIKZ_LEFT_KET(qubit))
        else:
            self.diagram.extend_lines_to_common_edge(self.diagram.qubits, offset=1)
        self.working_instructions = instructions
        self.index = 0
        while self.index < len(self.working_instructions):
            instr = self.working_instructions[self.index]
            if isinstance(instr, Pragma) and instr.command == PRAGMA_BEGIN_GROUP:
                self._build_group()
            elif isinstance(instr, Pragma) and instr.command == PRAGMA_END_GROUP:
                raise ValueError('PRAGMA {} found without matching {}.'.format(PRAGMA_END_GROUP, PRAGMA_BEGIN_GROUP))
            elif isinstance(instr, Measurement):
                self._build_measure()
            elif isinstance(instr, Gate):
                if 'FORKED' in instr.modifiers:
                    raise ValueError('LaTeX output does not currently supportFORKED modifiers: {}.'.format(instr))
                if len(instr.qubits) == 1:
                    self._build_1q_unitary()
                elif instr.name in SOURCE_TARGET_OP and (not instr.modifiers):
                    self._build_custom_source_target_op()
                else:
                    self._build_generic_unitary()
            elif isinstance(instr, UNSUPPORTED_INSTRUCTION_CLASSES):
                raise ValueError('LaTeX output does not currently supportthe following instruction: {}'.format(instr.out()))
            else:
                self.index += 1
        self.diagram.extend_lines_to_common_edge(self.diagram.qubits)
        self.index = 0
        self.working_instructions = measures
        for _ in self.working_instructions:
            self._build_measure()
        offset = max(self.settings.qubit_line_open_wire_length, 0)
        self.diagram.extend_lines_to_common_edge(self.diagram.qubits, offset=offset)
        return self.diagram

    def _build_group(self) -> None:
        """
        Update the partial diagram with the subcircuit delimited by the grouping PRAGMA.

        Advances the index beyond the ending pragma.
        """
        assert self.working_instructions is not None
        instr = self.working_instructions[self.index]
        assert isinstance(instr, Pragma)
        if len(instr.args) != 0:
            raise ValueError(f'PRAGMA {PRAGMA_BEGIN_GROUP} expected a freeform string, or nothing at all.')
        start = self.index + 1
        for j in range(start, len(self.working_instructions)):
            instruction_j = self.working_instructions[j]
            if isinstance(instruction_j, Pragma) and instruction_j.command == PRAGMA_END_GROUP:
                block_settings = replace(self.settings, label_qubit_lines=False, qubit_line_open_wire_length=0)
                subcircuit = Program(*self.working_instructions[start:j])
                block = DiagramBuilder(subcircuit, block_settings).build()
                block_name = instr.freeform_string if instr.freeform_string else ''
                assert self.diagram is not None
                self.diagram.append_diagram(block, group=block_name)
                self.index = j + 1
                return
        raise ValueError('Unable to find PRAGMA {} matching {}.'.format(PRAGMA_END_GROUP, instr))

    def _build_measure(self) -> None:
        """
        Update the partial diagram with a measurement operation.

        Advances the index by one.
        """
        assert self.working_instructions is not None
        instr = self.working_instructions[self.index]
        assert isinstance(instr, Measurement)
        assert self.diagram is not None
        self.diagram.append(instr.qubit.index, TIKZ_MEASURE())
        self.index += 1

    def _build_custom_source_target_op(self) -> None:
        """
        Update the partial diagram with a single operation involving a source and a target
        (e.g. a controlled gate, a swap).

        Advances the index by one.
        """
        assert self.working_instructions is not None
        instr = self.working_instructions[self.index]
        assert isinstance(instr, Gate)
        source, target = qubit_indices(instr)
        assert self.diagram is not None
        displaced = self.diagram.interval(min(source, target), max(source, target))
        self.diagram.extend_lines_to_common_edge(displaced)
        source_op, target_op = SOURCE_TARGET_OP[instr.name]
        offset = (-1 if source > target else 1) * (len(displaced) - 1)
        self.diagram.append(source, source_op(source, offset))
        self.diagram.append(target, target_op())
        self.diagram.extend_lines_to_common_edge(displaced)
        self.index += 1

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