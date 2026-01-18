from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING
from cirq import circuits, ops
def _is_temp(q: 'cirq.Qid') -> bool:
    return isinstance(q, (ops.CleanQubit, ops.BorrowableQubit))