from typing import Iterator, Callable, TYPE_CHECKING
import sympy
from cirq import ops
from cirq.interop.quirk.cells.cell import CellMaker
from cirq.interop.quirk.cells.parse import parse_formula
def _formula_gate(identifier: str, default_formula: str, gate_func: Callable[['cirq.TParamVal'], 'cirq.Gate']) -> CellMaker:
    return CellMaker(identifier=identifier, size=gate_func(0).num_qubits(), maker=lambda args: gate_func(parse_formula(default_formula if args.value is None else args.value)).on(*args.qubits))