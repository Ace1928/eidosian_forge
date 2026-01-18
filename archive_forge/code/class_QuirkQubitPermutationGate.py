from typing import Callable, Iterator, Sequence, Tuple, TYPE_CHECKING
from cirq import ops, value
from cirq.interop.quirk.cells.cell import CELL_SIZES, CellMaker
@value.value_equality
class QuirkQubitPermutationGate(ops.QubitPermutationGate):
    """A qubit permutation gate specified by a permutation list."""

    def __init__(self, identifier: str, name: str, permutation: Sequence[int]):
        """Inits QuirkQubitPermutationGate.

        Args:
            identifier: Quirk identifier string.
            name: Label to include in circuit diagram info.
            permutation: A shuffled sequence of integers from 0 to
                len(permutation) - 1. The entry at offset `i` is the result
                of permuting `i`.
        """
        self.identifier = identifier
        self.name = name
        super().__init__(permutation)

    def _value_equality_values_(self):
        return (self.identifier, self.name, self.permutation)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> Tuple[str, ...]:
        return tuple((f'{self.name}[{i}>{self.permutation[i]}]' for i in range(len(self.permutation))))

    def __repr__(self) -> str:
        return f'cirq.interop.quirk.QuirkQubitPermutationGate(identifier={repr(self.identifier)},name={repr(self.name)},permutation={repr(self.permutation)})'