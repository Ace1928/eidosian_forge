import itertools
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.transpiler.passes.optimization.template_matching.forward_match import ForwardMatch
from qiskit.transpiler.passes.optimization.template_matching.backward_match import BackwardMatch
def _sublist(self, lst, exclude, length):
    """
        Function that returns all possible combinations of a given length, considering an
        excluded list of elements.
        Args:
            lst (list): list of qubits indices from the circuit.
            exclude (list): list of qubits from the first matched circuit gate.
            length (int): length of the list to be returned (number of template qubit -
            number of qubit from the first matched template gate).
        Yield:
            iterator: Iterator of the possible lists.
        """
    for sublist in itertools.combinations([e for e in lst if e not in exclude], length):
        yield list(sublist)