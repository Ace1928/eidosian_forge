from typing import Collection, Dict, Optional, List, Set, Tuple, cast
import cirq
from cirq_google.api import v2
from cirq_google.api.v2 import device_pb2
from cirq_google.devices import grid_device
from cirq_google.experimental.ops import coupler_pulse
from cirq_google.ops import physical_z_tag, sycamore_gate
def _parse_device(s: str) -> Tuple[List[cirq.GridQubit], Dict[str, Set[cirq.GridQubit]]]:
    """Parse ASCIIart device layout into info about qubits and connectivity.

    Args:
        s: String representing the qubit layout. Each line represents a row,
            and each character in the row is a qubit, or a blank site if the
            character is a hyphen '-'. Different letters for the qubit specify
            which measurement line that qubit is connected to, e.g. all 'A'
            qubits share a measurement line. Leading and trailing spaces on
            each line are ignored.

    Returns:
        A list of qubits and a dict mapping measurement line name to the qubits
        on that measurement line.
    """
    lines = s.strip().split('\n')
    qubits: List[cirq.GridQubit] = []
    measurement_lines: Dict[str, Set[cirq.GridQubit]] = {}
    for row, line in enumerate(lines):
        for col, c in enumerate(line.strip()):
            if c != '-':
                qubit = cirq.GridQubit(row, col)
                qubits.append(qubit)
                measurement_line = measurement_lines.setdefault(c, set())
                measurement_line.add(qubit)
    return (qubits, measurement_lines)