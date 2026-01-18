from functools import reduce
from typing import List, NamedTuple, Sequence, Tuple
from dataclasses import dataclass
import numpy as np
import cirq
from cirq import value
from cirq._compat import proper_repr, proper_eq
from cirq.transformers.heuristic_decompositions.gate_tabulation_math_utils import (
class TwoQubitGateTabulationResult(NamedTuple):
    """Represents a compilation of a target 2-qubit with respect to a base
    gate.

    This object encodes the relationship between 4x4 unitary operators

    U_target ~ k_N · U_base · k_{N-1} · ... · k_1 · U_base · k_0

    where U_target, U_base are 2-local and k_j are 1-local.

    Attributes:
        base_gate: 4x4 unitary denoting U_base above.
        target_gate: 4x4 unitary denoting U_target above.
        local_unitaries: Sequence of 2-tuples
            $(k_{00}, k_{01}), (k_{10}, k_{11}) \\ldots$ where
            $k_j = k_{j0} \\otimes k_{j1}$ in the product above.
            Each $k_{j0}, k_{j1}$ is a 2x2 unitary.
        actual_gate: 4x4 unitary denoting the right hand side above, ideally
            equal to U_target.
        success: Whether actual_gate is expected to be close to U_target.
    """
    base_gate_unitary: np.ndarray
    target_gate: np.ndarray
    local_unitaries: Tuple[_SingleQubitGatePair, ...]
    actual_gate: np.ndarray
    success: bool