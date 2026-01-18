from __future__ import annotations
from math import pi, inf, isclose
from typing import Any
from copy import deepcopy
from itertools import product
from functools import partial
import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap, Target
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.synthesis.one_qubit import one_qubit_decompose
from qiskit.synthesis.two_qubit.xx_decompose import XXDecomposer, XXEmbodiments
from qiskit.synthesis.two_qubit.two_qubit_decompose import (
from qiskit.quantum_info import Operator
from qiskit.circuit import ControlFlowOp, Gate, Parameter
from qiskit.circuit.library.standard_gates import (
from qiskit.transpiler.passes.synthesis import plugin
from qiskit.transpiler.passes.optimization.optimize_1q_decomposition import (
from qiskit.providers.models import BackendProperties
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.exceptions import QiskitError
def _run_main_loop(self, dag, qubit_indices, plugin_method, plugin_kwargs, default_method, default_kwargs):
    """Inner loop for the optimizer, after all DAG-independent set-up has been completed."""
    for node in dag.op_nodes(ControlFlowOp):
        node.op = node.op.replace_blocks([dag_to_circuit(self._run_main_loop(circuit_to_dag(block), {inner: qubit_indices[outer] for inner, outer in zip(block.qubits, node.qargs)}, plugin_method, plugin_kwargs, default_method, default_kwargs), copy_operations=False) for block in node.op.blocks])
    for node in dag.named_nodes(*self._synth_gates):
        if self._min_qubits is not None and len(node.qargs) < self._min_qubits:
            continue
        synth_dag = None
        unitary = node.op.to_matrix()
        n_qubits = len(node.qargs)
        if plugin_method.max_qubits is not None and n_qubits > plugin_method.max_qubits or (plugin_method.min_qubits is not None and n_qubits < plugin_method.min_qubits):
            method, kwargs = (default_method, default_kwargs)
        else:
            method, kwargs = (plugin_method, plugin_kwargs)
        if method.supports_coupling_map:
            kwargs['coupling_map'] = (self._coupling_map, [qubit_indices[x] for x in node.qargs])
        synth_dag = method.run(unitary, **kwargs)
        if synth_dag is not None:
            dag.substitute_node_with_dag(node, synth_dag)
    return dag