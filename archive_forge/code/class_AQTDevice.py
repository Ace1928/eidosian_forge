import json
from typing import Any, cast, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import networkx as nx
import numpy as np
import cirq
from cirq_aqt import aqt_device_metadata
@cirq.value_equality
class AQTDevice(cirq.Device):
    """Ion trap device with qubits having all-to-all connectivity and placed on a line."""

    def __init__(self, measurement_duration: 'cirq.DURATION_LIKE', twoq_gates_duration: 'cirq.DURATION_LIKE', oneq_gates_duration: 'cirq.DURATION_LIKE', qubits: Iterable[cirq.LineQubit]) -> None:
        """Initializes the description of an ion trap device.

        Args:
            measurement_duration: The maximum duration of a measurement.
            twoq_gates_duration: The maximum duration of a two qubit operation.
            oneq_gates_duration: The maximum duration of a single qubit
            operation.
            qubits: Qubits on the device, identified by their x location.

        Raises:
            TypeError: If not all the qubits supplied are `cirq.LineQubit`s.
        """
        if not all((isinstance(qubit, cirq.LineQubit) for qubit in qubits)):
            raise TypeError(f'All qubits were not of type cirq.LineQubit, instead were {set((type(qubit) for qubit in qubits))}')
        self.qubits = frozenset(qubits)
        graph = nx.Graph()
        graph.add_edges_from([(a, b) for a in qubits for b in qubits if a != b], directed=False)
        self._metadata = aqt_device_metadata.AQTDeviceMetadata(qubits=self.qubits, measurement_duration=measurement_duration, twoq_gates_duration=twoq_gates_duration, oneq_gates_duration=oneq_gates_duration)

    @property
    def metadata(self) -> aqt_device_metadata.AQTDeviceMetadata:
        return self._metadata

    def validate_gate(self, gate: cirq.Gate):
        if gate not in self.metadata.gateset:
            raise ValueError(f'Unsupported gate type: {gate!r}')

    def validate_operation(self, operation):
        if not isinstance(operation, cirq.GateOperation):
            raise ValueError(f'Unsupported operation: {operation!r}')
        self.validate_gate(operation.gate)
        for q in operation.qubits:
            if not isinstance(q, cirq.LineQubit):
                raise ValueError(f'Unsupported qubit type: {q!r}')
            if q not in self.qubits:
                raise ValueError(f'Qubit not on device: {q!r}')

    def validate_circuit(self, circuit: cirq.AbstractCircuit):
        super().validate_circuit(circuit)
        _verify_unique_measurement_keys(circuit.all_operations())

    def at(self, position: int) -> Optional[cirq.LineQubit]:
        """Returns the qubit at the given position, if there is one, else None."""
        q = cirq.LineQubit(position)
        return q if q in self.qubits else None

    def _value_equality_values_(self) -> Any:
        return (self.metadata, self.qubits)

    def __str__(self) -> str:
        diagram = cirq.TextDiagramDrawer()
        for q in self.qubits:
            diagram.write(q.x, 0, str(q))
            for q2 in q.neighbors(self.qubits):
                diagram.grid_line(q.x, 0, q2.x, 0)
        return diagram.render(horizontal_spacing=3, vertical_spacing=2, use_unicode_characters=True)

    def __repr__(self) -> str:
        return f'cirq_aqt.aqt_device.AQTDevice(measurement_duration={self.metadata.measurement_duration!r}, twoq_gates_duration={self.metadata.twoq_gates_duration!r}, oneq_gates_duration={self.metadata.oneq_gates_duration!r}, qubits={sorted(self.qubits)!r})'

    def _repr_pretty_(self, p: Any, cycle: bool):
        """iPython (Jupyter) pretty print."""
        p.text('AQTDevice(...)' if cycle else self.__str__())