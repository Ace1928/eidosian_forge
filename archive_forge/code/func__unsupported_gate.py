from typing import Iterator
from cirq.interop.quirk.cells.cell import CellMaker, CELL_SIZES
def _unsupported_gate(identifier: str, reason: str) -> CellMaker:

    def fail(_):
        raise NotImplementedError(f'Converting the Quirk gate {identifier} is not implemented yet. Reason: {reason}')
    return CellMaker(identifier, 0, fail)