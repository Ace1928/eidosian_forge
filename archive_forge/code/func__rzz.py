from typing import Iterator, List, Optional
import itertools
import math
import numpy as np
import cirq
from cirq_google import ops
def _rzz(theta: float, q0: cirq.Qid, q1: cirq.Qid) -> cirq.OP_TREE:
    """Implements the Rzz Ising coupling gate (i.e. exp(-1j * theta * zz)) using Sycamore gates.

    Args:
        theta: The rotation parameter of Rzz Ising coupling gate.
        q0: First qubit to operate on
        q1: Second qubit to operate on

    Yields:
        The `cirq.OP_TREE` that implements the Rzz Ising coupling gate using Sycamore gates.
    """
    phi = -np.pi / 24
    c_phi = np.cos(2 * phi)
    target_unitary = cirq.unitary(cirq.ZZPowGate(exponent=2 * theta / np.pi, global_shift=-0.5))
    c2 = abs(np.sin(theta) if abs(np.cos(theta)) > c_phi else np.cos(theta)) / c_phi
    program = cirq.Circuit(ops.SYC(q0, q1), cirq.rx(2 * np.arccos(c2)).on(q1), ops.SYC(q0, q1))
    yield _create_corrected_circuit(target_unitary, program, q0, q1)