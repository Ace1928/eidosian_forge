import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford, Pauli
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
def _calc_decoupling(pauli_x, pauli_z, qubit_list, min_qubit, num_qubits, cliff):
    """Calculate a decoupling operator D:
    D^{-1} * Ox * D = x1
    D^{-1} * Oz * D = z1
    and reduce the clifford such that it will act trivially on min_qubit
    """
    circ = QuantumCircuit(num_qubits)
    decouple_cliff = cliff.copy()
    num_qubits = decouple_cliff.num_qubits
    decouple_cliff.phase = np.zeros(2 * num_qubits)
    decouple_cliff.symplectic_matrix = np.eye(2 * num_qubits)
    qubit0 = min_qubit
    for qubit in qubit_list:
        typeq = _from_pair_paulis_to_type(pauli_x, pauli_z, qubit)
        if typeq in [[[True, True], [False, False]], [[True, True], [True, True]], [[True, True], [True, False]]]:
            circ.s(qubit)
            _append_s(decouple_cliff, qubit)
        elif typeq in [[[True, False], [False, False]], [[True, False], [True, False]], [[True, False], [False, True]], [[False, False], [False, True]]]:
            circ.h(qubit)
            _append_h(decouple_cliff, qubit)
        elif typeq in [[[False, False], [True, True]], [[True, False], [True, True]]]:
            circ.s(qubit)
            circ.h(qubit)
            _append_s(decouple_cliff, qubit)
            _append_h(decouple_cliff, qubit)
        elif typeq == [[True, True], [False, True]]:
            circ.h(qubit)
            circ.s(qubit)
            _append_h(decouple_cliff, qubit)
            _append_s(decouple_cliff, qubit)
        elif typeq == [[False, True], [True, True]]:
            circ.s(qubit)
            circ.h(qubit)
            circ.s(qubit)
            _append_s(decouple_cliff, qubit)
            _append_h(decouple_cliff, qubit)
            _append_s(decouple_cliff, qubit)
    A_qubits = []
    B_qubits = []
    C_qubits = []
    D_qubits = []
    for qubit in qubit_list:
        typeq = _from_pair_paulis_to_type(pauli_x, pauli_z, qubit)
        if typeq in A_class:
            A_qubits.append(qubit)
        elif typeq in B_class:
            B_qubits.append(qubit)
        elif typeq in C_class:
            C_qubits.append(qubit)
        elif typeq in D_class:
            D_qubits.append(qubit)
    if len(A_qubits) % 2 != 1:
        raise QiskitError('Symplectic Gaussian elimination fails.')
    if qubit0 not in A_qubits:
        qubitA = A_qubits[0]
        circ.swap(qubit0, qubitA)
        _append_swap(decouple_cliff, qubit0, qubitA)
        if qubit0 in B_qubits:
            B_qubits.remove(qubit0)
            B_qubits.append(qubitA)
            A_qubits.remove(qubitA)
            A_qubits.append(qubit0)
        elif qubit0 in C_qubits:
            C_qubits.remove(qubit0)
            C_qubits.append(qubitA)
            A_qubits.remove(qubitA)
            A_qubits.append(qubit0)
        elif qubit0 in D_qubits:
            D_qubits.remove(qubit0)
            D_qubits.append(qubitA)
            A_qubits.remove(qubitA)
            A_qubits.append(qubit0)
        else:
            A_qubits.remove(qubitA)
            A_qubits.append(qubit0)
    for qubit in C_qubits:
        circ.cx(qubit0, qubit)
        _append_cx(decouple_cliff, qubit0, qubit)
    for qubit in D_qubits:
        circ.cx(qubit, qubit0)
        _append_cx(decouple_cliff, qubit, qubit0)
    if len(B_qubits) > 1:
        for qubit in B_qubits[1:]:
            qubitB = B_qubits[0]
            circ.cx(qubitB, qubit)
            _append_cx(decouple_cliff, qubitB, qubit)
    if len(B_qubits) > 0:
        qubitB = B_qubits[0]
        circ.cx(qubit0, qubitB)
        circ.h(qubitB)
        circ.cx(qubitB, qubit0)
        _append_cx(decouple_cliff, qubit0, qubitB)
        _append_h(decouple_cliff, qubitB)
        _append_cx(decouple_cliff, qubitB, qubit0)
    Alen = int((len(A_qubits) - 1) / 2)
    if Alen > 0:
        A_qubits.remove(qubit0)
    for qubit in range(Alen):
        circ.cx(A_qubits[2 * qubit + 1], A_qubits[2 * qubit])
        circ.cx(A_qubits[2 * qubit], qubit0)
        circ.cx(qubit0, A_qubits[2 * qubit + 1])
        _append_cx(decouple_cliff, A_qubits[2 * qubit + 1], A_qubits[2 * qubit])
        _append_cx(decouple_cliff, A_qubits[2 * qubit], qubit0)
        _append_cx(decouple_cliff, qubit0, A_qubits[2 * qubit + 1])
    return (circ, decouple_cliff)