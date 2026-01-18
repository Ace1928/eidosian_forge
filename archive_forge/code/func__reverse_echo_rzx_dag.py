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
def _reverse_echo_rzx_dag(theta):
    """Return the following circuit

        .. parsed-literal::

                 ┌───┐┌───────────────┐     ┌────────────────┐┌───┐
            q_0: ┤ H ├┤1              ├─────┤1               ├┤ H ├─────
                 ├───┤│  Rzx(theta/2) │┌───┐│  Rzx(-theta/2) │├───┤┌───┐
            q_1: ┤ H ├┤0              ├┤ X ├┤0               ├┤ X ├┤ H ├
                 └───┘└───────────────┘└───┘└────────────────┘└───┘└───┘
        """
    reverse_rzx_dag = DAGCircuit()
    qr = QuantumRegister(2)
    reverse_rzx_dag.add_qreg(qr)
    reverse_rzx_dag.apply_operation_back(HGate(), [qr[0]], [])
    reverse_rzx_dag.apply_operation_back(HGate(), [qr[1]], [])
    reverse_rzx_dag.apply_operation_back(RZXGate(theta / 2), [qr[1], qr[0]], [])
    reverse_rzx_dag.apply_operation_back(XGate(), [qr[1]], [])
    reverse_rzx_dag.apply_operation_back(RZXGate(-theta / 2), [qr[1], qr[0]], [])
    reverse_rzx_dag.apply_operation_back(XGate(), [qr[1]], [])
    reverse_rzx_dag.apply_operation_back(HGate(), [qr[0]], [])
    reverse_rzx_dag.apply_operation_back(HGate(), [qr[1]], [])
    return reverse_rzx_dag