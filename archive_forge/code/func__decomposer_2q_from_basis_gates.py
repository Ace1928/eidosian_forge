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
def _decomposer_2q_from_basis_gates(basis_gates, pulse_optimize=None, approximation_degree=None):
    decomposer2q = None
    kak_gate = _choose_kak_gate(basis_gates)
    euler_basis = _choose_euler_basis(basis_gates)
    basis_fidelity = approximation_degree or 1.0
    if isinstance(kak_gate, RZXGate):
        backup_optimizer = TwoQubitBasisDecomposer(CXGate(), basis_fidelity=basis_fidelity, euler_basis=euler_basis, pulse_optimize=pulse_optimize)
        decomposer2q = XXDecomposer(euler_basis=euler_basis, backup_optimizer=backup_optimizer)
    elif kak_gate is not None:
        decomposer2q = TwoQubitBasisDecomposer(kak_gate, basis_fidelity=basis_fidelity, euler_basis=euler_basis, pulse_optimize=pulse_optimize)
    return decomposer2q