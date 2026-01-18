import heapq
from qiskit.circuit.controlledgate import ControlledGate
def _init_matched_blocked_list(self):
    """
        Initialize the list of blocked and matchedwith attributes.
        Returns:
            Tuple[list, list, list, list]:
            First list contains the attributes matchedwith in the circuit,
            second list contains the attributes isblocked in the circuit,
            third list contains the attributes matchedwith in the template,
            fourth list contains the attributes isblocked in the template.
        """
    circuit_matched = []
    circuit_blocked = []
    for node in self.circuit_dag_dep.get_nodes():
        circuit_matched.append(node.matchedwith)
        circuit_blocked.append(node.isblocked)
    template_matched = []
    template_blocked = []
    for node in self.template_dag_dep.get_nodes():
        template_matched.append(node.matchedwith)
        template_blocked.append(node.isblocked)
    return (circuit_matched, circuit_blocked, template_matched, template_blocked)