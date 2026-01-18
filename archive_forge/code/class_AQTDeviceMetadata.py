from typing import Any, Iterable, Mapping
import networkx as nx
import cirq
from cirq_aqt import aqt_target_gateset
class AQTDeviceMetadata(cirq.DeviceMetadata):
    """Hardware metadata for ion trap device with all-connected qubits placed on a line."""

    def __init__(self, qubits: Iterable['cirq.LineQubit'], measurement_duration: 'cirq.DURATION_LIKE', twoq_gates_duration: 'cirq.DURATION_LIKE', oneq_gates_duration: 'cirq.DURATION_LIKE'):
        """Create metadata object for AQTDevice.

        Args:
            qubits: Iterable of `cirq.LineQubit`s that exist on the device.
            measurement_duration: The maximum duration of a measurement.
            twoq_gates_duration: The maximum duration of a two qubit operation.
            oneq_gates_duration: The maximum duration of a single qubit operation.
        """
        graph = nx.Graph()
        graph.add_edges_from([(a, b) for a in qubits for b in qubits if a != b], directed=False)
        super().__init__(qubits, graph)
        self._gateset = aqt_target_gateset.AQTTargetGateset()
        self._measurement_duration = cirq.Duration(measurement_duration)
        self._twoq_gates_duration = cirq.Duration(twoq_gates_duration)
        self._oneq_gates_duration = cirq.Duration(oneq_gates_duration)
        self._gate_durations = {cirq.GateFamily(cirq.MeasurementGate): self._measurement_duration, cirq.GateFamily(cirq.XXPowGate): self._twoq_gates_duration, cirq.GateFamily(cirq.XPowGate): self._oneq_gates_duration, cirq.GateFamily(cirq.YPowGate): self._oneq_gates_duration, cirq.GateFamily(cirq.ZPowGate): self._oneq_gates_duration, cirq.GateFamily(cirq.PhasedXPowGate): self._oneq_gates_duration}
        assert not self._gateset.gates.symmetric_difference(self._gate_durations.keys()), 'AQTDeviceMetadata.gate_durations must have the same Gates as AQTTargetGateset.'

    @property
    def gateset(self) -> 'cirq.Gateset':
        """Returns the `cirq.Gateset` of supported gates on this device."""
        return self._gateset

    @property
    def gate_durations(self) -> Mapping['cirq.GateFamily', 'cirq.Duration']:
        """Get a dictionary of supported gate families and their gate operation durations.

        Use `duration_of` to obtain duration of a specific `cirq.GateOperation` instance.
        """
        return self._gate_durations

    @property
    def measurement_duration(self) -> 'cirq.DURATION_LIKE':
        """Return the maximum duration of the measurement operation."""
        return self._measurement_duration

    @property
    def oneq_gates_duration(self) -> 'cirq.DURATION_LIKE':
        """Return the maximum duration of an operation on one-qubit gates."""
        return self._oneq_gates_duration

    @property
    def twoq_gates_duration(self) -> 'cirq.DURATION_LIKE':
        """Return the maximum duration of an operation on two-qubit gates."""
        return self._twoq_gates_duration

    def duration_of(self, operation: 'cirq.Operation') -> 'cirq.DURATION_LIKE':
        """Return the maximum duration of the specified gate operation.

        Args:
            operation: The `cirq.Operation` for which to determine its duration.

        Raises:
            ValueError: if the operation has an unsupported gate type.
        """
        for gate_family, duration in self.gate_durations.items():
            if operation in gate_family:
                return duration
        else:
            raise ValueError(f'Unsupported gate type: {operation!r}')

    def _value_equality_values_(self) -> Any:
        return (self._measurement_duration, self._twoq_gates_duration, self._oneq_gates_duration, self.qubit_set)

    def __repr__(self) -> str:
        return f'cirq_aqt.aqt_device_metadata.AQTDeviceMetadata(qubits={sorted(self.qubit_set)!r}, measurement_duration={self.measurement_duration!r}, twoq_gates_duration={self.twoq_gates_duration!r}, oneq_gates_duration={self.oneq_gates_duration!r})'