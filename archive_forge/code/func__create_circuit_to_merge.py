from typing import Optional, List
import pytest
import cirq
from cirq.transformers.transformer_primitives import MAPPED_CIRCUIT_OP_TAG
def _create_circuit_to_merge():
    q = cirq.LineQubit.range(3)
    return cirq.Circuit(cirq.Moment(cirq.H.on_each(*q)), cirq.CNOT(q[0], q[2]), cirq.CNOT(*q[0:2]), cirq.H(q[0]), cirq.CZ(*q[:2]), cirq.X(q[0]), cirq.Y(q[1]), cirq.CNOT(*q[0:2]), cirq.CNOT(*q[1:3]), cirq.X(q[0]), cirq.Moment(cirq.X(q[0]).with_tags('ignore'), cirq.Y(q[1])), cirq.CNOT(*q[:2]), strategy=cirq.InsertStrategy.NEW)