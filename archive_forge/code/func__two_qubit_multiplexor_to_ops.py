from typing import Union, Tuple, Sequence, List, Optional
import numpy as np
import cirq
from cirq import ops
from cirq import transformers as opt
def _two_qubit_multiplexor_to_ops(q0: ops.Qid, q1: ops.Qid, q2: ops.Qid, u1: np.ndarray, u2: np.ndarray, shift_left: bool=True, diagonal: Optional[np.ndarray]=None, atol: float=1e-08) -> Tuple[Optional[np.ndarray], List[ops.Operation]]:
    """Converts a two qubit double multiplexor to circuit.
    Input: U_1 ⊕ U_2, with select qubit a (i.e. a = |0> => U_1(b,c),
    a = |1> => U_2(b,c).

    We want this:
        $$
        U_1 ⊕ U_2 = (V ⊕ V) @ (D ⊕ D^{\\dagger}) @ (W ⊕ W)
        $$
    We can get it via:
        $$
        U_1 = V @ D @ W       (1)
        U_2 = V @ D^{\\dagger} @ W (2)
        $$

    We can derive
        $$
        U_1 U_2^{\\dagger}= V @ D^2 @ V^{\\dagger}, (3)
        $$

    i.e the eigendecomposition of $U_1 U_2^{\\dagger}$ will give us D and V.
    W is easy to derive from (2).

    This function, after calculating V, D and W, also returns the circuit that
    implements these unitaries: V, W on qubits b, c and the middle diagonal
    multiplexer on a,b,c qubits.

    The resulting circuit will have only known two-qubit and one-qubit gates,
    namely CZ, CNOT and rx, ry, PhasedXPow gates.

    Args:
        q0: first qubit
        q1: second qubit
        q2: third qubit
        u1: two-qubit operation on b,c for a = |0>
        u2: two-qubit operation on b,c for a = |1>
        shift_left: return the extracted diagonal or not
        diagonal: an incoming diagonal to be merged with
        atol: the absolute tolerance for the two-qubit sub-decompositions.

    Returns:
        The circuit implementing the two qubit multiplexor consisting only of
        known two-qubit and single qubit gates
    """
    u1u2 = u1 @ u2.conj().T
    eigvals, v = cirq.unitary_eig(u1u2)
    d = np.diag(np.sqrt(eigvals))
    w = d @ v.conj().T @ u2
    circuit_u1u2_mid = _middle_multiplexor_to_ops(q0, q1, q2, eigvals)
    if diagonal is not None:
        v = diagonal @ v
    d_v, circuit_u1u2_r = opt.two_qubit_matrix_to_diagonal_and_cz_operations(q1, q2, v, atol=atol)
    w = d_v @ w
    d_w: Optional[np.ndarray]
    if shift_left:
        d_w, circuit_u1u2_l = opt.two_qubit_matrix_to_diagonal_and_cz_operations(q1, q2, w, atol=atol)
    else:
        d_w = None
        circuit_u1u2_l = opt.two_qubit_matrix_to_cz_operations(q1, q2, w, allow_partial_czs=False, atol=atol)
    return (d_w, circuit_u1u2_l + circuit_u1u2_mid + circuit_u1u2_r)