from qiskit.circuit.controlledgate import ControlledGate
def _update_successor(self, node, successor_id):
    """
        Return a node with an updated attribute 'SuccessorToVisit'.
        Args:
            node (DAGDepNode): current node.
            successor_id (int): successor id to remove.

        Returns:
            DAGOpNode or DAGOutNode: Node with updated attribute 'SuccessorToVisit'.
        """
    node_update = node
    node_update.successorstovisit.pop(successor_id)
    return node_update