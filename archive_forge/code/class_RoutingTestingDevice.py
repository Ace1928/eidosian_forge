from typing import TYPE_CHECKING
import networkx as nx
from cirq import devices, ops
class RoutingTestingDevice(devices.Device):
    """Testing device to be used for testing qubit connectivity in routing procedures."""

    def __init__(self, nx_graph: nx.Graph) -> None:
        relabeling_map = {old: ops.q(old) if isinstance(old, (int, str)) else ops.q(*old) for old in nx_graph}
        nx.relabel_nodes(nx_graph, relabeling_map, copy=False)
        self._metadata = devices.DeviceMetadata(relabeling_map.values(), nx_graph)

    @property
    def metadata(self) -> devices.DeviceMetadata:
        return self._metadata

    def validate_operation(self, operation: 'cirq.Operation') -> None:
        if not self._metadata.qubit_set.issuperset(operation.qubits):
            raise ValueError(f'Qubits not on device: {operation.qubits!r}.')
        if len(operation.qubits) > 1:
            if len(operation.qubits) == 2:
                if operation.qubits not in self._metadata.nx_graph.edges:
                    raise ValueError(f'Qubit pair is not a valid edge on device: {operation.qubits!r}.')
                return
            if not isinstance(operation.gate, ops.MeasurementGate):
                raise ValueError(f'Unsupported operation: {operation}. Routing device only supports 1 / 2 qubit operations.')