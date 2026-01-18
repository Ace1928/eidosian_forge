from qiskit.circuit.controlledgate import ControlledGate
def _update_qarg_indices(self, qarg):
    """
        Change qubits indices of the current circuit node in order to
        be comparable with the indices of the template qubits list.
        Args:
            qarg (list): list of qubits indices from the circuit for a given node.
        """
    self.qarg_indices = []
    for q in qarg:
        if q in self.qubits:
            self.qarg_indices.append(self.qubits.index(q))
    if len(qarg) != len(self.qarg_indices):
        self.qarg_indices = []