from typing import Tuple
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library.standard_gates import RZXGate, HGate, XGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes.calibration.rzx_builder import _check_calibration_type, CRCalType
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
Run the EchoRZXWeylDecomposition pass on `dag`.

        Rewrites two-qubit gates in an arbitrary circuit in terms of echoed cross-resonance
        gates by computing the Weyl decomposition of the corresponding unitary. Modifies the
        input dag.

        Args:
            dag (DAGCircuit): DAG to rewrite.

        Returns:
            DAGCircuit: The modified dag.

        Raises:
            TranspilerError: If the circuit cannot be rewritten.
        