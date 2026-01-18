from qiskit.circuit.delay import Delay
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import AnalysisPass
Run duration validation passes.

        Args:
            dag: DAG circuit to check instruction durations.
        