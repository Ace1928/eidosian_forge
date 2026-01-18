from typing import Any, TypeVar, Optional, Sequence, Union
import numpy as np
from typing_extensions import Protocol
from cirq._doc import doc_private
from cirq.protocols import unitary_protocol
def _strat_distance_from_unitary(val: Any) -> Optional[float]:
    """Attempts to compute a value's trace_distance_bound from its unitary."""
    u = unitary_protocol.unitary(val, default=None)
    if u is None:
        return NotImplemented
    if u.shape == (2, 2):
        squared = 1 - (0.5 * abs(u[0][0] + u[1][1])) ** 2
        if squared <= 0:
            return 0.0
        return squared ** 0.5
    return trace_distance_from_angle_list(np.angle(np.linalg.eigvals(u)))