from __future__ import annotations
from collections.abc import Callable
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford  # pylint: disable=cyclic-import
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
from qiskit.synthesis.linear import (
from qiskit.synthesis.linear_phase import synth_cz_depth_line_mr, synth_cx_cz_depth_line_my
from qiskit.synthesis.linear.linear_matrix_utils import (
def _create_graph_state(cliff, validate=False):
    """Given a Clifford cliff (denoted by U) that induces a stabilizer state U |0>,
    apply a layer H1 of Hadamard gates to a subset of the qubits to make H1 U |0> into a graph state,
    namely to make cliff.stab_x matrix have full rank.
    Returns the QuantumCircuit H1_circ that includes the Hadamard gates and the updated Clifford
    that induces the graph state.
    The algorithm is based on Lemma 6 in [2].

    Args:
        cliff (Clifford): a Clifford operator.
        validate (Boolean): if True, validates the synthesis process.

    Returns:
        H1_circ: a circuit containing a layer of Hadamard gates.
        cliffh: cliffh.stab_x has full rank.

    Raises:
        QiskitError: if there are errors in the Gauss elimination process.

    References:
        2. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
           Phys. Rev. A 70, 052328 (2004).
           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_
    """
    num_qubits = cliff.num_qubits
    rank = _compute_rank(cliff.stab_x)
    H1_circ = QuantumCircuit(num_qubits, name='H1')
    cliffh = cliff.copy()
    if rank < num_qubits:
        stab = cliff.stab[:, :-1]
        stab = _gauss_elimination(stab, num_qubits)
        Cmat = stab[rank:num_qubits, num_qubits:]
        Cmat = np.transpose(Cmat)
        Cmat, perm = _gauss_elimination_with_perm(Cmat)
        perm = perm[0:num_qubits - rank]
        if validate:
            if _compute_rank(Cmat) != num_qubits - rank:
                raise QiskitError('The matrix Cmat after Gauss elimination has wrong rank.')
            if _compute_rank(stab[:, 0:num_qubits]) != rank:
                raise QiskitError('The matrix after Gauss elimination has wrong rank.')
            for i in range(rank, num_qubits):
                if stab[i, 0:num_qubits].any():
                    raise QiskitError('After Gauss elimination, the final num_qubits - rank rowscontain non-zero elements')
        for qubit in perm:
            H1_circ.h(qubit)
            _append_h(cliffh, qubit)
        if validate:
            stabh = cliffh.stab_x
            if _compute_rank(stabh) != num_qubits:
                raise QiskitError('The state is not a graph state.')
    return (H1_circ, cliffh)