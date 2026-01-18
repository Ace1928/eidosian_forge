import abc
from typing import Any, Dict, Sequence, Tuple, TYPE_CHECKING
from typing_extensions import Self
from cirq import protocols
from cirq.ops import pauli_string as ps, raw_types
class PauliStringGateOperation(raw_types.Operation, metaclass=abc.ABCMeta):

    def __init__(self, pauli_string: ps.PauliString) -> None:
        self._pauli_string = pauli_string

    @property
    def pauli_string(self) -> 'cirq.PauliString':
        return self._pauli_string

    def validate_args(self, qubits: Sequence[raw_types.Qid]) -> None:
        if len(qubits) != len(self.pauli_string):
            raise ValueError('Incorrect number of qubits for gate')

    def with_qubits(self, *new_qubits: 'cirq.Qid') -> Self:
        self.validate_args(new_qubits)
        return self.map_qubits(dict(zip(self.pauli_string.qubits, new_qubits)))

    @abc.abstractmethod
    def map_qubits(self, qubit_map: Dict[raw_types.Qid, raw_types.Qid]) -> Self:
        """Return an equivalent operation on new qubits with its Pauli string
        mapped to new qubits.

        new_pauli_string = self.pauli_string.map_qubits(qubit_map)
        """

    @property
    def qubits(self) -> Tuple[raw_types.Qid, ...]:
        return tuple(self.pauli_string)

    def _pauli_string_diagram_info(self, args: 'protocols.CircuitDiagramInfoArgs', exponent: Any=1) -> 'cirq.CircuitDiagramInfo':
        qubits = self.qubits if args.known_qubits is None else args.known_qubits
        syms = tuple((f'[{self.pauli_string[qubit]}]' for qubit in qubits))
        return protocols.CircuitDiagramInfo(wire_symbols=syms, exponent=exponent)