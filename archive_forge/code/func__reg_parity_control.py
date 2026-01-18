from typing import Any, Iterable, Iterator, List, Optional, TYPE_CHECKING, Union
from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker
def _reg_parity_control(identifier: str, *, basis_change: Optional['cirq.Gate']=None) -> CellMaker:
    return CellMaker(identifier=identifier, size=1, maker=lambda args: ParityControlCell(qubits=args.qubits, basis_change=_basis_else_empty(basis_change, args.qubits)))