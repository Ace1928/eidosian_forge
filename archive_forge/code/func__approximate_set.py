import math
import warnings
from functools import lru_cache
from scipy.spatial import KDTree
import pennylane as qml
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript
@lru_cache()
def _approximate_set(basis_gates, max_length=10):
    """Builds an approximate unitary set required for the `Solovay-Kitaev algorithm <https://arxiv.org/abs/quant-ph/0505030>`_.

    Args:
        basis_gates (list(str)): Basis set to be used for Solovay-Kitaev decomposition build using
            following terms, ``['X', 'Y', 'Z', 'H', 'T', 'T*', 'S', 'S*']``, where `*` refers
            to the gate adjoint.
        max_length (int): Maximum expansion length of Clifford+T sequences in the approximation set. Default is `10`

    Returns:
        Tuple(list[list[~pennylane.operation.Operation]], list[TensorLike], list[float], list[TensorLike]): A tuple containing the list of
        Clifford+T sequences that will be used for approximating a matrix in the base case of recursive implementation of
        Solovay-Kitaev algorithm, with their corresponding SU(2) representations, global phases, and quaternion representations.
    """
    _CLIFFORD_T_BASIS = {'I': qml.Identity(0), 'X': qml.X(0), 'Y': qml.Y(0), 'Z': qml.Z(0), 'H': qml.Hadamard(0), 'T': qml.T(0), 'T*': qml.adjoint(qml.T(0)), 'S': qml.S(0), 'S*': qml.adjoint(qml.S(0))}
    basis = [_CLIFFORD_T_BASIS[gate.upper()] for gate in basis_gates]
    basis_mat, basis_gph = ({}, {})
    for gate in basis:
        su2_mat, su2_gph = _SU2_transform(gate.matrix())
        basis_mat.update({gate: su2_mat})
        basis_gph.update({gate: su2_gph})
    gtrie_ids = [[[gate] for gate in basis]]
    gtrie_mat = [list(basis_mat.values())]
    gtrie_gph = [list(basis_gph.values())]
    approx_set_ids = list(gtrie_ids[0])
    approx_set_mat = list(gtrie_mat[0])
    approx_set_gph = list(gtrie_gph[0])
    approx_set_qat = [_quaternion_transform(mat) for mat in approx_set_mat]
    for depth in range(max_length - 1):
        gtrie_id, gtrie_mt, gtrie_gp = ([], [], [])
        for node, su2m, gphase in zip(gtrie_ids[depth], gtrie_mat[depth], gtrie_gph[depth]):
            last_op = qml.adjoint(node[-1], lazy=False) if node else None
            for op in basis:
                if qml.equal(op, last_op):
                    continue
                su2_gp = basis_gph[op] + gphase
                su2_op = (-1.0) ** bool(su2_gp >= math.pi) * (basis_mat[op] @ su2m)
                exists, quaternion = _contains_SU2(su2_op, approx_set_qat)
                if not exists:
                    approx_set_ids.append(node + [op])
                    approx_set_mat.append(su2_op)
                    approx_set_qat.append(quaternion)
                    gtrie_id.append(node + [op])
                    gtrie_mt.append(su2_op)
                    global_phase = qml.math.mod(su2_gp, math.pi)
                    approx_set_gph.append(global_phase)
                    gtrie_gp.append(global_phase)
        gtrie_ids.append(gtrie_id)
        gtrie_mat.append(gtrie_mt)
        gtrie_gph.append(gtrie_gp)
    return (approx_set_ids, approx_set_mat, approx_set_gph, approx_set_qat)