from typing import Any, cast, Optional, Type, Union
from cirq.ops import gateset, raw_types, parallel_gate, eigen_gate
from cirq import protocols
class AnyUnitaryGateFamily(gateset.GateFamily):
    """GateFamily which accepts any N-Qubit unitary gate."""

    def __init__(self, num_qubits: Optional[int]=None) -> None:
        """Init AnyUnitaryGateFamily

        Args:
            num_qubits: The GateFamily will accept any unitary gate acting on `num_qubits`.
                        If left `None`, the GateFamily will accept every unitary gate.
        Raises:
            ValueError: If `num_qubits` <= 0.
        """
        if num_qubits is not None and num_qubits <= 0:
            raise ValueError(f'num_qubits: {num_qubits} must be a positive integer.')
        self._num_qubits = num_qubits
        name = f'{(str(num_qubits) if num_qubits else 'Any')}-Qubit UnitaryGateFamily'
        kind = f'{num_qubits}-qubit ' if num_qubits else ''
        description = f'Accepts any {kind}unitary gate'
        super().__init__(raw_types.Gate, name=name, description=description)

    def _predicate(self, g: raw_types.Gate) -> bool:
        return (self._num_qubits is None or protocols.num_qubits(g) == self._num_qubits) and protocols.has_unitary(g)

    def __repr__(self) -> str:
        return f'cirq.AnyUnitaryGateFamily(num_qubits = {self._num_qubits})'

    def _value_equality_values_(self) -> Any:
        return self._num_qubits

    def _json_dict_(self):
        return {'num_qubits': self._num_qubits}

    @classmethod
    def _from_json_dict_(cls, num_qubits, **kwargs):
        return cls(num_qubits)