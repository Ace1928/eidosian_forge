from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import cirq
from cirq import ops, linalg, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker
@value.value_equality
class InputRotationCell(Cell):
    """Applies an operation that depends on an input gate."""

    def __init__(self, identifier: str, register: Optional[Sequence['cirq.Qid']], base_operation: 'cirq.Operation', exponent_sign: int):
        self.identifier = identifier
        self.register = None if register is None else tuple(register)
        self.base_operation = base_operation
        self.exponent_sign = exponent_sign

    def _value_equality_values_(self) -> Any:
        return (self.identifier, self.register, self.base_operation, self.exponent_sign)

    def __repr__(self) -> str:
        return f'cirq.interop.quirk.cells.input_rotation_cells.InputRotationCell(\n    {self.identifier!r},\n    {self.register!r},\n    {self.base_operation!r},\n    {self.exponent_sign!r})'

    def gate_count(self) -> int:
        return 1

    def with_line_qubits_mapped_to(self, qubits: List['cirq.Qid']) -> 'Cell':
        return InputRotationCell(self.identifier, None if self.register is None else Cell._replace_qubits(self.register, qubits), self.base_operation.with_qubits(*Cell._replace_qubits(self.base_operation.qubits, qubits)), exponent_sign=self.exponent_sign)

    def with_input(self, letter: str, register: Union[Sequence['cirq.Qid'], int]) -> 'Cell':
        if self.register is None and letter == 'a':
            if isinstance(register, int):
                raise ValueError('Dependent operation requires known length input; classical constant not allowed.')
            return InputRotationCell(self.identifier, register, self.base_operation, self.exponent_sign)
        return self

    def controlled_by(self, qubit: 'cirq.Qid'):
        return InputRotationCell(self.identifier, self.register, self.base_operation.controlled_by(qubit), self.exponent_sign)

    def operations(self) -> 'cirq.OP_TREE':
        if self.register is None:
            raise ValueError("Missing input 'a'")
        return QuirkInputRotationOperation(self.identifier, self.register, self.base_operation, self.exponent_sign)