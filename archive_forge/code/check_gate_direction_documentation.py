from qiskit.circuit import ControlFlowOp
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import AnalysisPass
Run the CheckGateDirection pass on `dag`.

        If `dag` is mapped and the direction is correct the property
        `is_direction_mapped` is set to True (or to False otherwise).

        Args:
            dag (DAGCircuit): DAG to check.
        