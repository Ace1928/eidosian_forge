from __future__ import annotations
from qiskit.circuit import Instruction, ParameterExpression, Qubit, Clbit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit.equivalence_library import EquivalenceLibrary
from qiskit.exceptions import QiskitError
from qiskit.transpiler import Target
from qiskit.transpiler.basepasses import TransformationPass
from .basis_translator import BasisTranslator
def _instruction_to_dag(op: Instruction) -> DAGCircuit:
    dag = DAGCircuit()
    dag.add_qubits([Qubit() for _ in range(op.num_qubits)])
    dag.add_qubits([Clbit() for _ in range(op.num_clbits)])
    dag.apply_operation_back(op, dag.qubits, dag.clbits, check=False)
    return dag