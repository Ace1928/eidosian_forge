from typing import Tuple
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library.standard_gates import RZXGate, HGate, XGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes.calibration.rzx_builder import _check_calibration_type, CRCalType
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
@staticmethod
def _echo_rzx_dag(theta):
    """Return the following circuit

        .. parsed-literal::

                 ┌───────────────┐┌───┐┌────────────────┐┌───┐
            q_0: ┤0              ├┤ X ├┤0               ├┤ X ├
                 │  Rzx(theta/2) │└───┘│  Rzx(-theta/2) │└───┘
            q_1: ┤1              ├─────┤1               ├─────
                 └───────────────┘     └────────────────┘
        """
    rzx_dag = DAGCircuit()
    qr = QuantumRegister(2)
    rzx_dag.add_qreg(qr)
    rzx_dag.apply_operation_back(RZXGate(theta / 2), [qr[0], qr[1]], [])
    rzx_dag.apply_operation_back(XGate(), [qr[0]], [])
    rzx_dag.apply_operation_back(RZXGate(-theta / 2), [qr[0], qr[1]], [])
    rzx_dag.apply_operation_back(XGate(), [qr[0]], [])
    return rzx_dag