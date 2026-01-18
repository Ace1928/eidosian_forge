import itertools
import logging
from math import inf
import numpy as np
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.circuit.classical import expr, types
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target
from qiskit.circuit import (
from qiskit._accelerate import stochastic_swap as stochastic_swap_rs
from qiskit._accelerate import nlayout
from qiskit.transpiler.passes.layout import disjoint_utils
from .utils import get_swap_map_dag
def _dag_from_block(block, node, root_dag):
    """Get a :class:`DAGCircuit` that represents the :class:`.QuantumCircuit` ``block`` embedded
    within the ``root_dag`` for full-width routing purposes.  This means that all the qubits are in
    the output DAG, but only the necessary clbits and classical registers are."""
    out = DAGCircuit()
    for qreg in root_dag.qregs.values():
        out.add_qreg(qreg)
    out.add_clbits(node.cargs)
    dummy = out.apply_operation_back(Instruction('dummy', len(node.qargs), len(node.cargs), []), node.qargs, node.cargs, check=False)
    wire_map = dict(itertools.chain(zip(block.qubits, node.qargs), zip(block.clbits, node.cargs)))
    out.substitute_node_with_dag(dummy, circuit_to_dag(block), wires=wire_map)
    return out