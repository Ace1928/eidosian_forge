from math import pi
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.circuit import QuantumRegister, ControlFlowOp
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit.library.standard_gates import (
@staticmethod
def _ryy_dag(parameter):
    _ryy_dag = DAGCircuit()
    qr = QuantumRegister(2)
    _ryy_dag.add_qreg(qr)
    _ryy_dag.apply_operation_back(RYYGate(parameter), [qr[1], qr[0]], [])
    return _ryy_dag