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
def _build_gate_lengths(props=None, target=None):
    """Builds a ``gate_lengths`` dictionary from either ``props`` (BackendV1)
    or ``target`` (BackendV2).

    The dictionary has the form:
    {gate_name: {(qubits,): duration}}
    """
    gate_lengths = {}
    if target is not None:
        for gate, prop_dict in target.items():
            gate_lengths[gate] = {}
            for qubit, gate_props in prop_dict.items():
                if gate_props is not None and gate_props.duration is not None:
                    gate_lengths[gate][qubit] = gate_props.duration
    elif props is not None:
        for gate in props._gates:
            gate_lengths[gate] = {}
            for k, v in props._gates[gate].items():
                length = v.get('gate_length')
                if length:
                    gate_lengths[gate][k] = length[0]
            if not gate_lengths[gate]:
                del gate_lengths[gate]
    return gate_lengths