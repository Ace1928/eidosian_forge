from typing import Iterator, List, Optional
import itertools
import math
import numpy as np
import cirq
from cirq_google import ops
def _decompose_cphase_into_syc(theta: float, q0: cirq.Qid, q1: cirq.Qid) -> cirq.OP_TREE:
    """Implements a cphase using the Ising gate generated from 2 Sycamore gates.

    A cphase gate has the matrix diag([1, 1, 1, exp(1j * theta)]) and can be mapped to the Rzz
    Ising gate +  single qubit Z rotations. We drop the global phase shift of theta / 4.

    Args:
        theta: The phase to apply, exp(1j * theta).
        q0: First qubit to operate on.
        q1: Second qubit to operate on.

    Yields:
        A `cirq.OP_TREE` implementing the cphase gate using Sycamore gates.
    """
    yield _rzz(-theta / 4, q0, q1)
    yield cirq.rz(theta / 2).on(q0)
    yield cirq.rz(theta / 2).on(q1)