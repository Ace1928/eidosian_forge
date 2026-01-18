from qiskit.circuit.controlledgate import ControlledGate
def _update_carg_indices(self, carg):
    """
        Change clbits indices of the current circuit node in order to
        be comparable with the indices of the template qubits list.
        Args:
            carg (list): list of clbits indices from the circuit for a given node.
        """
    self.carg_indices = []
    if carg:
        for q in carg:
            if q in self.clbits:
                self.carg_indices.append(self.clbits.index(q))
        if len(carg) != len(self.carg_indices):
            self.carg_indices = []