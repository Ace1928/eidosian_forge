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
def _handle_control_flow_ops(self, dag):
    """
        This is similar to transpiler/passes/utils/control_flow.py except that the
        collect blocks is redone for the control flow blocks.
        """
    pass_manager = PassManager()
    if 'run_list' in self.property_set:
        pass_manager.append(Collect1qRuns())
    if 'block_list' in self.property_set:
        pass_manager.append(Collect2qBlocks())
    pass_manager.append(self)
    for node in dag.op_nodes(ControlFlowOp):
        node.op = node.op.replace_blocks((pass_manager.run(block) for block in node.op.blocks))
    return dag