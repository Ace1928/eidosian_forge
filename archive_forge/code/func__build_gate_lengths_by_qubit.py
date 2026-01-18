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
def _build_gate_lengths_by_qubit(props=None, target=None):
    """
    Builds a `gate_lengths` dictionary from either `props` (BackendV1)
    or `target (BackendV2)`.

    The dictionary has the form:
    {(qubits): [Gate, duration]}
    """
    gate_lengths = {}
    if target is not None and target.qargs is not None:
        for qubits in target.qargs:
            names = target.operation_names_for_qargs(qubits)
            operation_and_durations = []
            for name in names:
                operation = target.operation_from_name(name)
                duration = getattr(target[name].get(qubits, None), 'duration', None)
                if duration:
                    operation_and_durations.append((operation, duration))
            if operation_and_durations:
                gate_lengths[qubits] = operation_and_durations
    elif props is not None:
        for gate_name, gate_props in props._gates.items():
            gate = GateNameToGate[gate_name]
            for qubits, properties in gate_props.items():
                duration = properties.get('gate_length', [0.0])[0]
                operation_and_durations = (gate, duration)
                if qubits in gate_lengths:
                    gate_lengths[qubits].append(operation_and_durations)
                else:
                    gate_lengths[qubits] = [operation_and_durations]
    return gate_lengths