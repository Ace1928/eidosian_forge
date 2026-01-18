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
def _choose_kak_gate(basis_gates):
    """Choose the first available 2q gate to use in the KAK decomposition."""
    kak_gate = None
    kak_gates = set(basis_gates or []).intersection(KAK_GATE_NAMES.keys())
    if kak_gates:
        kak_gate = KAK_GATE_NAMES[kak_gates.pop()]
    return kak_gate