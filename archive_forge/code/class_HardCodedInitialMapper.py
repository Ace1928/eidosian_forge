from typing import TYPE_CHECKING, Dict
import abc
from cirq import value
@value.value_equality
class HardCodedInitialMapper(AbstractInitialMapper):
    """Initial Mapper class takes a hard-coded mapping and returns it."""

    def __init__(self, _map: Dict['cirq.Qid', 'cirq.Qid']) -> None:
        self._map = _map

    def initial_mapping(self, circuit: 'cirq.AbstractCircuit') -> Dict['cirq.Qid', 'cirq.Qid']:
        """Returns the hard-coded initial mapping.

        Args:
            circuit: the input circuit with logical qubits.

        Returns:
            the hard-codded initial mapping.

        Raises:
            ValueError: if the qubits in circuit are not a subset of the qubit keys in the mapping.
        """
        if not circuit.all_qubits().issubset(set(self._map.keys())):
            raise ValueError('The qubits in circuit must be a subset of the keys in the mapping')
        return self._map

    def _value_equality_values_(self):
        return tuple(sorted(self._map.items()))

    def __repr__(self) -> str:
        return f'cirq.HardCodedInitialMapper({self._map})'