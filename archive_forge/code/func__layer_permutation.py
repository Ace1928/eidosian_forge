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
def _layer_permutation(self, dag, layer_partition, layout, qubit_subset, coupling, trials):
    """Find a swap circuit that implements a permutation for this layer.

        The goal is to swap qubits such that qubits in the same two-qubit gates
        are adjacent.

        Based on S. Bravyi's algorithm.

        Args:
            layer_partition (list): The layer_partition is a list of (qu)bit
                lists and each qubit is a tuple (qreg, index).
            layout (Layout): The layout is a Layout object mapping virtual
                qubits in the input circuit to physical qubits in the coupling
                graph. It reflects the current positions of the data.
            qubit_subset (list): The qubit_subset is the set of qubits in
                the coupling graph that we have chosen to map into, as tuples
                (Register, index).
            coupling (CouplingMap): Directed graph representing a coupling map.
                This coupling map should be one that was provided to the
                stochastic mapper.
            trials (int): Number of attempts the randomized algorithm makes.

        Returns:
            Tuple: success_flag, best_circuit, best_depth, best_layout

        If success_flag is True, then best_circuit contains a DAGCircuit with
        the swap circuit, best_depth contains the depth of the swap circuit,
        and best_layout contains the new positions of the data qubits after the
        swap circuit has been applied.

        Raises:
            TranspilerError: if anything went wrong.
        """
    logger.debug('layer_permutation: layer_partition = %s', layer_partition)
    logger.debug('layer_permutation: layout = %s', layout.get_virtual_bits())
    logger.debug('layer_permutation: qubit_subset = %s', qubit_subset)
    logger.debug('layer_permutation: trials = %s', trials)
    canonical_register = QuantumRegister(len(layout), 'q')
    gates = []
    for gate_args in layer_partition:
        if len(gate_args) > 2:
            raise TranspilerError('Layer contains > 2-qubit gates')
        if len(gate_args) == 2:
            gates.append(tuple(gate_args))
    logger.debug('layer_permutation: gates = %s', gates)
    dist = sum((coupling._dist_matrix[layout._v2p[g[0]], layout._v2p[g[1]]] for g in gates))
    logger.debug('layer_permutation: distance = %s', dist)
    if dist == len(gates):
        logger.debug('layer_permutation: nothing to do')
        circ = DAGCircuit()
        circ.add_qreg(canonical_register)
        return (True, circ, 0, layout)
    num_qubits = len(layout)
    best_depth = inf
    best_edges = None
    best_circuit = None
    best_layout = None
    cdist2 = coupling._dist_matrix ** 2
    int_qubit_subset = np.fromiter((dag.find_bit(bit).index for bit in qubit_subset), dtype=np.uint32, count=len(qubit_subset))
    int_gates = np.fromiter((dag.find_bit(bit).index for gate in gates for bit in gate), dtype=np.uint32, count=2 * len(gates))
    layout_mapping = {dag.find_bit(k).index: v for k, v in layout.get_virtual_bits().items()}
    int_layout = nlayout.NLayout(layout_mapping, num_qubits, coupling.size())
    trial_circuit = DAGCircuit()
    trial_circuit.add_qubits(layout.get_virtual_bits())
    edges = np.asarray(coupling.get_edges(), dtype=np.uint32).ravel()
    cdist = coupling._dist_matrix
    best_edges, best_layout, best_depth = stochastic_swap_rs.swap_trials(trials, num_qubits, int_layout, int_qubit_subset, int_gates, cdist, cdist2, edges, seed=self.seed)
    if best_layout is None:
        logger.debug('layer_permutation: failed!')
        return (False, None, None, None)
    edges = best_edges.edges()
    for idx in range(len(edges) // 2):
        swap_src = self._int_to_qubit[edges[2 * idx]]
        swap_tgt = self._int_to_qubit[edges[2 * idx + 1]]
        trial_circuit.apply_operation_back(SwapGate(), (swap_src, swap_tgt), (), check=False)
    best_circuit = trial_circuit
    logger.debug('layer_permutation: success!')
    layout_mapping = best_layout.layout_mapping()
    best_lay = Layout({best_circuit.qubits[k]: v for k, v in layout_mapping})
    return (True, best_circuit, best_depth, best_lay)