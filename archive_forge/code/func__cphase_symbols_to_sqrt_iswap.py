from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phxz
def _cphase_symbols_to_sqrt_iswap(a: 'cirq.Qid', b: 'cirq.Qid', turns: 'cirq.TParamVal', use_sqrt_iswap_inv: bool=True):
    """Implements `cirq.CZ(a, b) ** turns` using two √iSWAPs and single qubit rotations.

    Output unitary:
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, g]]
    where:
        g = exp(i·π·t).

    Args:
        a: The first qubit.
        b: The second qubit.
        turns: The rotational angle (t) that specifies the gate, where
            g = exp(i·π·t/2).
        use_sqrt_iswap_inv: If True, `cirq.SQRT_ISWAP_INV` is used instead of `cirq.SQRT_ISWAP`.

    Yields:
        A `cirq.OP_TREE` representing the decomposition.
    """
    theta = sympy.Mod(turns, 2.0) * sympy.pi
    sign = sympy.sign(sympy.pi - theta + 1e-09)
    theta_prime = sympy.pi - sign * sympy.pi + sign * theta
    phi = sympy.asin(np.sqrt(2) * sympy.sin(theta_prime / 4))
    xi = sympy.atan(sympy.tan(phi) / np.sqrt(2))
    yield ops.rz(sign * 0.5 * theta_prime).on(a)
    yield ops.rz(sign * 0.5 * theta_prime).on(b)
    yield ops.rx(xi).on(a)
    yield (ops.X(b) ** (-sign * 0.5))
    yield _sqrt_iswap_inv(a, b, use_sqrt_iswap_inv)
    yield ops.rx(-2 * phi).on(a)
    yield ops.Z(a)
    yield _sqrt_iswap_inv(a, b, use_sqrt_iswap_inv)
    yield ops.Z(a)
    yield ops.rx(xi).on(a)
    yield (ops.X(b) ** (sign * 0.5))