from qiskit.circuit.controlledgate import ControlledGate
def _is_same_op(self, node_circuit, node_template):
    """
        Check if two instructions are the same.
        Args:
            node_circuit (DAGDepNode): node in the circuit.
            node_template (DAGDepNode): node in the template.
        Returns:
            bool: True if the same, False otherwise.
        """
    return node_circuit.op.soft_compare(node_template.op)