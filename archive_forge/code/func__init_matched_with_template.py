from qiskit.circuit.controlledgate import ControlledGate
def _init_matched_with_template(self):
    """
        Initialize the attribute 'MatchedWith' in the circuit DAG dependency.
        """
    for i in range(0, self.template_dag_dep.size()):
        if i == self.node_id_t:
            self.template_dag_dep.get_node(i).matchedwith = [self.node_id_c]
        else:
            self.template_dag_dep.get_node(i).matchedwith = []