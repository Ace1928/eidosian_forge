from typing import Optional, Union, List, Tuple
import rustworkx as rx
from qiskit.circuit.operation import Operation
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import ControlFlowOp, ControlledGate, EquivalenceLibrary
from qiskit.transpiler.passes.utils import control_flow
from qiskit.transpiler.target import Target
from qiskit.transpiler.coupling import CouplingMap
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.routing.algorithms import ApproximateTokenSwapper
from qiskit.circuit.annotated_operation import (
from qiskit.synthesis.clifford import (
from qiskit.synthesis.linear import synth_cnot_count_full_pmh, synth_cnot_depth_line_kms
from qiskit.synthesis.permutation import (
from .plugin import HighLevelSynthesisPluginManager, HighLevelSynthesisPlugin
class BMSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Bravyi-Maslov method.

    The method only works on Cliffords with at most 3 qubits, for which it
    constructs the optimal CX cost decomposition.

    This plugin name is :``clifford.bm`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        if high_level_object.num_qubits <= 3:
            decomposition = synth_clifford_bm(high_level_object)
        else:
            decomposition = None
        return decomposition