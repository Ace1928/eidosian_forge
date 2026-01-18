from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.quantum_info import Operator
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.transpiler.basepasses import TransformationPass

        Run the CommutativeInverseCancellation pass on `dag`.

        Args:
            dag: the directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        