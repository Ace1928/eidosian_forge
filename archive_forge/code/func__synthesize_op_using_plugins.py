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
def _synthesize_op_using_plugins(self, op: Operation, qubits: List) -> Union[QuantumCircuit, None]:
    """
        Attempts to synthesize op using plugin mechanism.
        Returns either the synthesized circuit or None (which occurs when no
        synthesis methods are available or specified).
        """
    hls_plugin_manager = self.hls_plugin_manager
    if op.name in self.hls_config.methods.keys():
        methods = self.hls_config.methods[op.name]
    elif self.hls_config.use_default_on_unspecified and 'default' in hls_plugin_manager.method_names(op.name):
        methods = ['default']
    else:
        methods = []
    for method in methods:
        if isinstance(method, tuple):
            plugin_specifier, plugin_args = method
        else:
            plugin_specifier = method
            plugin_args = {}
        if isinstance(plugin_specifier, str):
            if plugin_specifier not in hls_plugin_manager.method_names(op.name):
                raise TranspilerError('Specified method: %s not found in available plugins for %s' % (plugin_specifier, op.name))
            plugin_method = hls_plugin_manager.method(op.name, plugin_specifier)
        else:
            plugin_method = plugin_specifier
        decomposition = plugin_method.run(op, coupling_map=self._coupling_map, target=self._target, qubits=qubits, **plugin_args)
        if decomposition is not None:
            return decomposition
    return None