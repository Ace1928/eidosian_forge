from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.circuit.library import PermutationGate
from qiskit.circuit.exceptions import CircuitError
class LinearFunctionsToPermutations(TransformationPass):
    """Promotes linear functions to permutations when possible."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the LinearFunctionsToPermutations pass on `dag`.
        Args:
            dag: input dag.
        Returns:
            Output dag with LinearFunctions synthesized.
        """
        for node in dag.named_nodes('linear_function'):
            try:
                pattern = node.op.permutation_pattern()
            except CircuitError:
                continue
            permutation = PermutationGate(pattern)
            dag.substitute_node(node, permutation)
        return dag