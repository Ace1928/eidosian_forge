from typing import Any, Iterable, Iterator, List, Optional, TYPE_CHECKING, Union
from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker
def _reg_control(identifier: str, *, basis_change: Optional['cirq.Gate']) -> CellMaker:
    return CellMaker(identifier=identifier, size=1, maker=lambda args: ControlCell(qubit=args.qubits[0], basis_change=_basis_else_empty(basis_change, args.qubits[0])))