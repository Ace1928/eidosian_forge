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
def _decompose_graph_state(cliff, validate, cz_synth_func):
    """Assumes that a stabilizer state of the Clifford cliff (denoted by U) corresponds to a graph state.
    Decompose it into the layers S1 - CZ1 - H2, such that:
    S1 CZ1 H2 |0> = U |0>,
    where S1_circ is a circuit that can contain only S gates,
    CZ1_circ is a circuit that can contain only CZ gates, and
    H2_circ is a circuit that can contain H gates on all qubits.

    Args:
        cliff (Clifford): a Clifford operator corresponding to a graph state, cliff.stab_x has full rank.
        validate (Boolean): if True, validates the synthesis process.
        cz_synth_func (Callable): a function to decompose the CZ sub-circuit.

    Returns:
        S1_circ: a circuit that can contain only S gates.
        CZ1_circ: a circuit that can contain only CZ gates.
        H2_circ: a circuit containing a layer of Hadamard gates.
        cliff_cpy: a Hadamard-free Clifford.

    Raises:
        QiskitError: if cliff does not induce a graph state.
    """
    num_qubits = cliff.num_qubits
    rank = _compute_rank(cliff.stab_x)
    cliff_cpy = cliff.copy()
    if rank < num_qubits:
        raise QiskitError('The stabilizer state is not a graph state.')
    S1_circ = QuantumCircuit(num_qubits, name='S1')
    H2_circ = QuantumCircuit(num_qubits, name='H2')
    stabx = cliff.stab_x
    stabz = cliff.stab_z
    stabx_inv = calc_inverse_matrix(stabx, validate)
    stabz_update = np.matmul(stabx_inv, stabz) % 2
    if validate:
        if (stabz_update != stabz_update.T).any():
            raise QiskitError('The multiplication of stabx_inv and stab_z is not a symmetric matrix.')
    CZ1_circ = cz_synth_func(stabz_update)
    for j in range(num_qubits):
        for i in range(0, j):
            if stabz_update[i][j]:
                _append_cz(cliff_cpy, i, j)
    for i in range(0, num_qubits):
        if stabz_update[i][i]:
            S1_circ.s(i)
            _append_s(cliff_cpy, i)
    for qubit in range(num_qubits):
        H2_circ.h(qubit)
        _append_h(cliff_cpy, qubit)
    return (H2_circ, CZ1_circ, S1_circ, cliff_cpy)