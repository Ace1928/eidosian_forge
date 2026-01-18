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
def _build_gate_errors_by_qubit(props=None, target=None):
    """
    Builds a `gate_error` dictionary from either `props` (BackendV1)
    or `target (BackendV2)`.

    The dictionary has the form:
    {(qubits): [Gate, error]}
    """
    gate_errors = {}
    if target is not None and target.qargs is not None:
        for qubits in target.qargs:
            names = target.operation_names_for_qargs(qubits)
            operation_and_errors = []
            for name in names:
                operation = target.operation_from_name(name)
                error = getattr(target[name].get(qubits, None), 'error', None)
                if error:
                    operation_and_errors.append((operation, error))
            if operation_and_errors:
                gate_errors[qubits] = operation_and_errors
    elif props is not None:
        for gate_name, gate_props in props._gates.items():
            gate = GateNameToGate[gate_name]
            for qubits, properties in gate_props.items():
                error = properties.get('gate_error', [0.0])[0]
                operation_and_errors = (gate, error)
                if qubits in gate_errors:
                    gate_errors[qubits].append(operation_and_errors)
                else:
                    gate_errors[qubits] = [operation_and_errors]
    return gate_errors