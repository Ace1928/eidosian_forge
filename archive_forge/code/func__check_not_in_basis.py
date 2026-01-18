from __future__ import annotations
import numpy as np
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.quantum_info import Operator
from qiskit.synthesis.two_qubit import TwoQubitBasisDecomposer
from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
from qiskit.circuit.library.standard_gates import CXGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.passes.synthesis import unitary_synthesis
from qiskit.transpiler.passes.utils import _block_to_matrix
from .collect_1q_runs import Collect1qRuns
from .collect_2q_blocks import Collect2qBlocks
def _check_not_in_basis(self, dag, gate_name, qargs):
    if self.target is not None:
        return not self.target.instruction_supported(gate_name, tuple((dag.find_bit(qubit).index for qubit in qargs)))
    else:
        return self.basis_gates and gate_name not in self.basis_gates