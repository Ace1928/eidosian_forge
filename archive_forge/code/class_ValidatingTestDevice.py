from typing import Tuple, AbstractSet, cast
from cirq import devices, ops
class ValidatingTestDevice(devices.Device):
    """A fake device that was created to ensure certain Device validation features are
    leveraged in Circuit functions. It contains the minimum set of features that tests
    require. Feel free to extend the features here as needed.

    Args:
        qubits: set of qubits on this device
        name: the name for repr
        allowed_gates: tuple of allowed gate types
        allowed_qubit_types: tuple of allowed qubit types
        validate_locality: if True, device will validate 2 qubit operations
            (except MeasurementGateOperations) whether the two qubits are adjacent. If True,
            GridQubits are assumed to be part of the allowed_qubit_types
        auto_decompose_gates: when set, for given gates it calls the cirq.decompose protocol
    """

    def __init__(self, qubits: AbstractSet[ops.Qid], name: str='ValidatingTestDevice', allowed_gates: Tuple[type, ...]=(ops.Gate,), allowed_qubit_types: Tuple[type, ...]=(devices.GridQubit,), validate_locality: bool=False, auto_decompose_gates: Tuple[type, ...]=tuple()):
        self.allowed_qubit_types = allowed_qubit_types
        self.allowed_gates = allowed_gates
        self.qubits = qubits
        self._repr = name
        self.validate_locality = validate_locality
        self.auto_decompose_gates = auto_decompose_gates
        if self.validate_locality and devices.GridQubit not in allowed_qubit_types:
            raise ValueError('GridQubit must be an allowed qubit type with validate_locality=True')

    def validate_operation(self, operation: ops.Operation) -> None:
        for q in operation.qubits:
            if not isinstance(q, self.allowed_qubit_types):
                raise ValueError(f'Unsupported qubit type: {type(q)!r}')
            if q not in self.qubits:
                raise ValueError(f'Qubit not on device: {q!r}')
        if not isinstance(operation.gate, self.allowed_gates):
            raise ValueError(f'Unsupported gate type: {operation.gate!r}')
        if self.validate_locality:
            if len(operation.qubits) == 2 and (not isinstance(operation.gate, ops.MeasurementGate)):
                p, q = operation.qubits
                if not cast(devices.GridQubit, p).is_adjacent(q):
                    raise ValueError(f'Non-local interaction: {operation!r}.')

    def __repr__(self):
        return self._repr