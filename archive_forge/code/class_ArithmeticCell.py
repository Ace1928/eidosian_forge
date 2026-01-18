import inspect
from typing import (
from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker, CELL_SIZES
@value.value_equality
class ArithmeticCell(Cell):

    def __init__(self, identifier: str, target: Sequence['cirq.Qid'], inputs: Sequence[Union[None, Sequence['cirq.Qid'], int]]):
        self.identifier = identifier
        self.target = tuple(target)
        self.inputs = tuple(inputs)

    def gate_count(self) -> int:
        return 1

    def _value_equality_values_(self) -> Any:
        return (self.identifier, self.target, self.inputs)

    def __repr__(self) -> str:
        return f'cirq.interop.quirk.cells.arithmetic_cells.ArithmeticCell(\n    {self.identifier!r},\n    {self.target!r},\n    {self.inputs!r})'

    def with_line_qubits_mapped_to(self, qubits: List['cirq.Qid']) -> 'Cell':
        return ArithmeticCell(identifier=self.identifier, target=Cell._replace_qubits(self.target, qubits), inputs=[e if e is None or isinstance(e, int) else Cell._replace_qubits(e, qubits) for e in self.inputs])

    @property
    def operation(self):
        return ARITHMETIC_OP_TABLE[self.identifier]

    def with_input(self, letter: str, register: Union[Sequence['cirq.Qid'], int]) -> 'ArithmeticCell':
        new_inputs = [reg if letter != reg_letter else register for reg, reg_letter in zip(self.inputs, self.operation.letters)]
        return ArithmeticCell(self.identifier, self.target, new_inputs)

    def operations(self) -> 'cirq.OP_TREE':
        missing_inputs = [letter for reg, letter in zip(self.inputs, self.operation.letters) if reg is None]
        if missing_inputs:
            raise ValueError(f'Missing input: {sorted(missing_inputs)}')
        inputs = cast(Sequence[Union[Sequence['cirq.Qid'], int]], self.inputs)
        qubits = self.target + tuple((q for i in self.inputs if isinstance(i, Sequence) for q in i))
        return QuirkArithmeticGate(self.identifier, [q.dimension for q in self.target], [i if isinstance(i, int) else [q.dimension for q in i] for i in inputs]).on(*qubits)