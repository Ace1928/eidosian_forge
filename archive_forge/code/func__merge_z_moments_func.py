from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def _merge_z_moments_func(m1: cirq.Moment, m2: cirq.Moment) -> Optional[cirq.Moment]:
    if any((op.gate != cirq.Z for m in [m1, m2] for op in m)):
        return None
    return cirq.Moment((cirq.Z(q) for q in m1.qubits | m2.qubits if m1.operates_on([q]) ^ m2.operates_on([q])))