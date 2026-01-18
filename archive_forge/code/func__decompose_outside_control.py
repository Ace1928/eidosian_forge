from typing import (
import numpy as np
import sympy
from cirq import linalg, protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import (
def _decompose_outside_control(self, control: 'cirq.Qid', near_target: 'cirq.Qid', far_target: 'cirq.Qid') -> 'cirq.OP_TREE':
    """A decomposition assuming one of the targets is in the middle.

        control: ───T──────@────────@───@────────────@────────────────
                           │        │   │            │
           near: ─X─T──────X─@─T^-1─X─@─X────@─X^0.5─X─@─X^0.5────────
                  │          │        │      │         │
            far: ─@─Y^-0.5─T─X─T──────X─T^-1─X─T^-1────X─S─────X^-0.5─
        """
    a, b, c = (control, near_target, far_target)
    t = common_gates.T
    sweep_abc = [common_gates.CNOT(a, b), common_gates.CNOT(b, c)]
    yield common_gates.CNOT(c, b)
    yield (pauli_gates.Y(c) ** (-0.5))
    yield (t(a), t(b), t(c))
    yield sweep_abc
    yield (t(b) ** (-1), t(c))
    yield sweep_abc
    yield (t(c) ** (-1))
    yield sweep_abc
    yield (t(c) ** (-1))
    yield (pauli_gates.X(b) ** 0.5)
    yield sweep_abc
    yield common_gates.S(c)
    yield (pauli_gates.X(b) ** 0.5)
    yield (pauli_gates.X(c) ** (-0.5))