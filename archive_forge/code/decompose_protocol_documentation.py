from typing import Any, FrozenSet, Sequence
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
Decomposes a value into operations, if possible.

    Args:
        val: The value to decompose into operations.

    Returns:
        A tuple of operations if decomposition succeeds.
    