import itertools
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.transpiler.passes.optimization.template_matching.forward_match import ForwardMatch
from qiskit.transpiler.passes.optimization.template_matching.backward_match import BackwardMatch
def _explore_circuit(self, node_id_c, node_id_t, n_qubits_t, length):
    """
        Explore the successors of the node_id_c (up to the given length).
        Args:
            node_id_c (int): first match id in the circuit.
            node_id_t (int): first match id in the template.
            n_qubits_t (int): number of qubits in the template.
            length (int): length for exploration of the successors.
        Returns:
            list: qubits configuration for the 'length' successors of node_id_c.
        """
    template_nodes = range(node_id_t + 1, self.template_dag_dep.size())
    circuit_nodes = range(0, self.circuit_dag_dep.size())
    successors_template = self.template_dag_dep.get_node(node_id_t).successors
    counter = 1
    qubit_set = set(self.circuit_dag_dep.get_node(node_id_c).qindices)
    if 2 * len(successors_template) > len(template_nodes):
        successors = self.circuit_dag_dep.get_node(node_id_c).successors
        for succ in successors:
            qarg = self.circuit_dag_dep.get_node(succ).qindices
            if len(qubit_set | set(qarg)) <= n_qubits_t and counter <= length:
                qubit_set = qubit_set | set(qarg)
                counter += 1
            elif len(qubit_set | set(qarg)) > n_qubits_t:
                return list(qubit_set)
        return list(qubit_set)
    else:
        not_successors = list(set(circuit_nodes) - set(self.circuit_dag_dep.get_node(node_id_c).successors))
        candidate = [not_successors[j] for j in range(len(not_successors) - 1, len(not_successors) - 1 - length, -1)]
        for not_succ in candidate:
            qarg = self.circuit_dag_dep.get_node(not_succ).qindices
            if counter <= length and len(qubit_set | set(qarg)) <= n_qubits_t:
                qubit_set = qubit_set | set(qarg)
                counter += 1
            elif len(qubit_set | set(qarg)) > n_qubits_t:
                return list(qubit_set)
        return list(qubit_set)