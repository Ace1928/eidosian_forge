import logging
import math
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.synthesis.one_qubit import one_qubit_decompose
from qiskit._accelerate import euler_one_qubit_decomposer
from qiskit.circuit.library.standard_gates import (
from qiskit.circuit import Qubit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
def _possible_decomposers(basis_set):
    decomposers = []
    if basis_set is None:
        decomposers = list(one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES)
    else:
        euler_basis_gates = one_qubit_decompose.ONE_QUBIT_EULER_BASIS_GATES
        for euler_basis_name, gates in euler_basis_gates.items():
            if set(gates).issubset(basis_set):
                decomposers.append(euler_basis_name)
        if 'U3' in decomposers and 'U321' in decomposers:
            decomposers.remove('U3')
        if 'ZSX' in decomposers and 'ZSXX' in decomposers:
            decomposers.remove('ZSX')
    return decomposers