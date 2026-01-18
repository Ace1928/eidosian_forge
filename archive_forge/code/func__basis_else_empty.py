from typing import Any, Iterable, Iterator, List, Optional, TYPE_CHECKING, Union
from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker
def _basis_else_empty(basis_change: Optional['cirq.Gate'], qureg: Union['cirq.Qid', Iterable['cirq.Qid']]) -> Iterable['cirq.Operation']:
    if basis_change is None:
        return ()
    return basis_change.on_each(qureg)