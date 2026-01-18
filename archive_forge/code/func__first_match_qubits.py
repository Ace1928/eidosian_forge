import copy
import itertools
from collections import OrderedDict
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.transforms import transform
from pennylane import adjoint
from pennylane.ops.qubit.attributes import symmetric_over_all_wires
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms.commutation_dag import commutation_dag
from pennylane.wires import Wires
def _first_match_qubits(node_c, node_p, n_qubits_p):
    """
    Returns the list of qubits for circuit given the first match, the unknown qubit are
    replaced by -1.
    Args:
        node_c (.CommutationDAGNode): First matched node in the circuit.
        node_p (.CommutationDAGNode): First matched node in the pattern.
        n_qubits_p (int): number of qubit in the pattern.
    Returns:
        list: list of qubits to consider in circuit (with specific order).
    """
    control_base = {'CNOT': 'PauliX', 'CZ': 'PauliZ', 'CCZ': 'PauliZ', 'CY': 'PauliY', 'CH': 'Hadamard', 'CSWAP': 'SWAP', 'Toffoli': 'PauliX', 'ControlledPhaseShift': 'PhaseShift', 'CRX': 'RX', 'CRY': 'RY', 'CRZ': 'RZ', 'CRot': 'Rot', 'MultiControlledX': 'PauliX', 'ControlledOperation': 'ControlledOperation'}
    first_match_qubits = []
    if len(node_c.op.control_wires) >= 1:
        circuit_control = node_c.op.control_wires
        circuit_target = Wires([w for w in node_c.op.wires if w not in node_c.op.control_wires])
        if control_base[node_p.op.name] not in symmetric_over_all_wires:
            for control_permuted in itertools.permutations(circuit_control):
                control_permuted = list(control_permuted)
                first_match_qubits_sub = [-1] * n_qubits_p
                for q in node_p.wires:
                    node_circuit_perm = control_permuted + circuit_target
                    first_match_qubits_sub[q] = node_circuit_perm[node_p.wires.index(q)]
                first_match_qubits.append(first_match_qubits_sub)
        else:
            for control_permuted in itertools.permutations(circuit_control):
                control_permuted = list(control_permuted)
                for target_permuted in itertools.permutations(circuit_target):
                    target_permuted = list(target_permuted)
                    first_match_qubits_sub = [-1] * n_qubits_p
                    for q in node_p.wires:
                        node_circuit_perm = control_permuted + target_permuted
                        first_match_qubits_sub[q] = node_circuit_perm[node_p.wires.index(q)]
                    first_match_qubits.append(first_match_qubits_sub)
    elif node_p.op.name not in symmetric_over_all_wires:
        first_match_qubits_sub = [-1] * n_qubits_p
        for q in node_p.wires:
            first_match_qubits_sub[q] = node_c.wires[node_p.wires.index(q)]
        first_match_qubits.append(first_match_qubits_sub)
    else:
        for perm_q in itertools.permutations(node_c.wires):
            first_match_qubits_sub = [-1] * n_qubits_p
            for q in node_p.wires:
                first_match_qubits_sub[q] = perm_q[node_p.wires.index(q)]
            first_match_qubits.append(first_match_qubits_sub)
    return first_match_qubits