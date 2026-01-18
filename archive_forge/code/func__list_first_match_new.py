import itertools
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.transpiler.passes.optimization.template_matching.forward_match import ForwardMatch
from qiskit.transpiler.passes.optimization.template_matching.backward_match import BackwardMatch
def _list_first_match_new(self, node_circuit, node_template, n_qubits_t, n_clbits_t):
    """
        Returns the list of qubit for circuit given the first match, the unknown qubit are
        replaced by -1.
        Args:
            node_circuit (DAGDepNode): First match node in the circuit.
            node_template (DAGDepNode): First match node in the template.
            n_qubits_t (int): number of qubit in the template.
            n_clbits_t (int): number of classical bit in the template.
        Returns:
            list: list of qubits to consider in circuit (with specific order).
        """
    l_q = []
    if isinstance(node_circuit.op, ControlledGate) and node_template.op.num_ctrl_qubits > 1:
        control = node_template.op.num_ctrl_qubits
        control_qubits_circuit = node_circuit.qindices[:control]
        not_control_qubits_circuit = node_circuit.qindices[control:]
        if node_template.op.base_gate.name not in ['rxx', 'ryy', 'rzz', 'swap', 'iswap', 'ms']:
            for control_perm_q in itertools.permutations(control_qubits_circuit):
                control_perm_q = list(control_perm_q)
                l_q_sub = [-1] * n_qubits_t
                for q in node_template.qindices:
                    node_circuit_perm = control_perm_q + not_control_qubits_circuit
                    l_q_sub[q] = node_circuit_perm[node_template.qindices.index(q)]
                l_q.append(l_q_sub)
        else:
            for control_perm_q in itertools.permutations(control_qubits_circuit):
                control_perm_q = list(control_perm_q)
                for not_control_perm_q in itertools.permutations(not_control_qubits_circuit):
                    not_control_perm_q = list(not_control_perm_q)
                    l_q_sub = [-1] * n_qubits_t
                    for q in node_template.qindices:
                        node_circuit_perm = control_perm_q + not_control_perm_q
                        l_q_sub[q] = node_circuit_perm[node_template.qindices.index(q)]
                    l_q.append(l_q_sub)
    elif node_template.op.name not in ['rxx', 'ryy', 'rzz', 'swap', 'iswap', 'ms']:
        l_q_sub = [-1] * n_qubits_t
        for q in node_template.qindices:
            l_q_sub[q] = node_circuit.qindices[node_template.qindices.index(q)]
        l_q.append(l_q_sub)
    else:
        for perm_q in itertools.permutations(node_circuit.qindices):
            l_q_sub = [-1] * n_qubits_t
            for q in node_template.qindices:
                l_q_sub[q] = perm_q[node_template.qindices.index(q)]
            l_q.append(l_q_sub)
    if not node_template.cindices or not node_circuit.cindices:
        l_c = []
    else:
        l_c = [-1] * n_clbits_t
        for c in node_template.cindices:
            l_c[c] = node_circuit[node_template.cindices.index(c)]
    return (l_q, l_c)