import itertools
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.transpiler.passes.optimization.template_matching.forward_match import ForwardMatch
from qiskit.transpiler.passes.optimization.template_matching.backward_match import BackwardMatch
def _list_qubit_clbit_circuit(self, list_first_match, permutation):
    """
        Function that returns the list of the circuit qubits and clbits give a permutation
        and an initial match.
        Args:
            list_first_match (list): list of qubits indices for the initial match.
            permutation (list): possible permutation for the circuit qubit.
        Returns:
            list: list of circuit qubit for the given permutation and initial match.
        """
    list_circuit = []
    counter = 0
    for elem in list_first_match:
        if elem == -1:
            list_circuit.append(permutation[counter])
            counter = counter + 1
        else:
            list_circuit.append(elem)
    return list_circuit