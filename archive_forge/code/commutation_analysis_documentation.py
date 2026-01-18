from collections import defaultdict
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.basepasses import AnalysisPass
Run the CommutationAnalysis pass on `dag`.

        Run the pass on the DAG, and write the discovered commutation relations
        into the ``property_set``.
        