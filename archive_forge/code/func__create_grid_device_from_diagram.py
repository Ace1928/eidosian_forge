from typing import Collection, Dict, Optional, List, Set, Tuple, cast
import cirq
from cirq_google.api import v2
from cirq_google.api.v2 import device_pb2
from cirq_google.devices import grid_device
from cirq_google.experimental.ops import coupler_pulse
from cirq_google.ops import physical_z_tag, sycamore_gate
def _create_grid_device_from_diagram(ascii_grid: str, gateset: cirq.Gateset, gate_durations: Optional[Dict['cirq.GateFamily', 'cirq.Duration']]=None) -> grid_device.GridDevice:
    """Parse ASCIIart device layout into a GridDevice instance.

    This function assumes that all pairs of adjacent qubits are valid targets
    for two-qubit gates.

    Args:
        ascii_grid: ASCII version of the grid (see _parse_device for details).
        gateset: The device's gate set.
        gate_durations: A map of durations for each gate in the gate set.
        out: If given, populate this proto, otherwise create a new proto.
    """
    qubits, _ = _parse_device(ascii_grid)
    qubit_set = frozenset(qubits)
    pairs: List[Tuple[cirq.GridQubit, cirq.GridQubit]] = []
    for qubit in qubits:
        for neighbor in sorted(qubit.neighbors()):
            if neighbor > qubit and neighbor in qubit_set:
                pairs.append((qubit, cast(cirq.GridQubit, neighbor)))
    return grid_device.GridDevice._from_device_information(qubit_pairs=pairs, gateset=gateset, gate_durations=gate_durations)