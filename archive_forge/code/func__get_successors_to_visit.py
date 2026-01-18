from qiskit.circuit.controlledgate import ControlledGate
def _get_successors_to_visit(self, node, list_id):
    """
        Return the successor for a given node and id.
        Args:
            node (DAGOpNode or DAGOutNode): current node.
            list_id (int): id in the list for the successor to get.

        Returns:
            int: id of the successor to get.
        """
    successor_id = node.successorstovisit[list_id]
    return successor_id