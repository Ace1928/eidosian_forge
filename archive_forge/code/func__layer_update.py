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
def _layer_update(self, dag, layer, best_layout, best_depth, best_circuit):
    """Add swaps followed by the now mapped layer from the original circuit.

        Args:
            dag (DAGCircuit): The DAGCircuit object that the _mapper method is building
            layer (DAGCircuit): A DAGCircuit layer from the original circuit
            best_layout (Layout): layout returned from _layer_permutation
            best_depth (int): depth returned from _layer_permutation
            best_circuit (DAGCircuit): swap circuit returned from _layer_permutation
        """
    logger.debug('layer_update: layout = %s', best_layout)
    logger.debug('layer_update: self.initial_layout = %s', self.initial_layout)
    if best_depth > 0:
        logger.debug('layer_update: there are swaps in this layer, depth %d', best_depth)
        dag.compose(best_circuit, qubits={bit: bit for bit in best_circuit.qubits})
    else:
        logger.debug('layer_update: there are no swaps in this layer')
    dag.compose(layer['graph'], qubits=best_layout.reorder_bits(dag.qubits))