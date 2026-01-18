from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.target import Target
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.converters import circuit_to_dag
Run the CheckMap pass on `dag`.

        If `dag` is mapped to `coupling_map`, the property
        `is_swap_mapped` is set to True (or to False otherwise).

        Args:
            dag (DAGCircuit): DAG to map.
        