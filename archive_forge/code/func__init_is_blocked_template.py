from qiskit.circuit.controlledgate import ControlledGate
def _init_is_blocked_template(self):
    """
        Initialize the attribute 'IsBlocked' in the template DAG dependency.
        """
    for i in range(0, self.template_dag_dep.size()):
        self.template_dag_dep.get_node(i).isblocked = False