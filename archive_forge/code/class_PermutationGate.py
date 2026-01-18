import abc
from typing import (
from cirq import circuits, ops, protocols, transformers, value
from cirq.type_workarounds import NotImplementedType
class PermutationGate(ops.Gate, metaclass=abc.ABCMeta):
    """A permutation gate indicates a change in the mapping from qubits to
    logical indices.

    Args:
        num_qubits: The number of qubits the gate should act on.
        swap_gate: The gate that swaps the indices mapped to by a pair of
            qubits (e.g. SWAP or fermionic swap).
    """

    def __init__(self, num_qubits: int, swap_gate: 'cirq.Gate'=ops.SWAP) -> None:
        self._num_qubits = num_qubits
        self.swap_gate = swap_gate

    def num_qubits(self) -> int:
        return self._num_qubits

    @abc.abstractmethod
    def permutation(self) -> Dict[int, int]:
        """permutation = {i: s[i]} indicates that the i-th element is mapped to
        the s[i]-th element."""

    def update_mapping(self, mapping: Dict[ops.Qid, LogicalIndex], keys: Sequence['cirq.Qid']) -> None:
        """Updates a mapping (in place) from qubits to logical indices.

        Args:
            mapping: The mapping to update.
            keys: The qubits acted on by the gate.
        """
        permutation = self.permutation()
        indices = tuple(permutation.keys())
        new_keys = [keys[permutation[i]] for i in indices]
        old_elements = [mapping.get(keys[i]) for i in indices]
        for new_key, old_element in zip(new_keys, old_elements):
            if old_element is None:
                if new_key in mapping:
                    del mapping[new_key]
            else:
                mapping[new_key] = old_element

    @staticmethod
    def validate_permutation(permutation: Dict[int, int], n_elements: Optional[int]=None) -> None:
        if not permutation:
            return
        if set(permutation.values()) != set(permutation):
            raise IndexError('key and value sets must be the same.')
        if min(permutation) < 0:
            raise IndexError('keys of the permutation must be non-negative.')
        if n_elements is not None:
            if max(permutation) >= n_elements:
                raise IndexError('key is out of bounds.')

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> Union[str, Iterable[str], 'cirq.CircuitDiagramInfo']:
        if args.known_qubit_count is None:
            return NotImplemented
        permutation = self.permutation()
        arrow = '↦' if args.use_unicode_characters else '->'
        wire_symbols = tuple((str(i) + arrow + str(permutation.get(i, i)) for i in range(self.num_qubits())))
        return wire_symbols