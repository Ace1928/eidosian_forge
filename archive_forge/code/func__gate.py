from typing import Iterator, Callable, TYPE_CHECKING
import sympy
from cirq import ops
from cirq.interop.quirk.cells.cell import CellMaker
from cirq.interop.quirk.cells.parse import parse_formula
def _gate(identifier: str, gate: 'cirq.Gate') -> CellMaker:
    return CellMaker(identifier=identifier, size=gate.num_qubits(), maker=lambda args: gate.on(*args.qubits))