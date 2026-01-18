import copy
import dataclasses
import logging
import functools
import time
import numpy as np
import rustworkx as rx
from qiskit.converters import dag_to_circuit
from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.passes.layout.set_layout import SetLayout
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
from qiskit.transpiler.passes.layout.apply_layout import ApplyLayout
from qiskit.transpiler.passes.layout import disjoint_utils
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit._accelerate.nlayout import NLayout
from qiskit._accelerate.sabre_layout import sabre_layout_and_routing
from qiskit._accelerate.sabre_swap import (
from qiskit.transpiler.passes.routing.sabre_swap import _build_sabre_dag, _apply_sabre_result
from qiskit.transpiler.target import Target
from qiskit.transpiler.coupling import CouplingMap
from qiskit.utils.parallel import CPU_COUNT
def _inner_run(self, dag, coupling_map, starting_layouts=None):
    if not coupling_map.is_symmetric:
        coupling_map = copy.deepcopy(coupling_map)
        coupling_map.make_symmetric()
    neighbor_table = NeighborTable(rx.adjacency_matrix(coupling_map.graph))
    dist_matrix = coupling_map.distance_matrix
    original_qubit_indices = {bit: index for index, bit in enumerate(dag.qubits)}
    partial_layouts = []
    if starting_layouts is not None:
        coupling_map_reverse_mapping = {coupling_map.graph[x]: x for x in coupling_map.graph.node_indices()}
        for layout in starting_layouts:
            virtual_bits = layout.get_virtual_bits()
            out_layout = [None] * len(dag.qubits)
            for bit, phys in virtual_bits.items():
                pos = original_qubit_indices.get(bit, None)
                if pos is None:
                    continue
                out_layout[pos] = coupling_map_reverse_mapping[phys]
            partial_layouts.append(out_layout)
    sabre_dag, circuit_to_dag_dict = _build_sabre_dag(dag, coupling_map.size(), original_qubit_indices)
    sabre_start = time.perf_counter()
    initial_layout, final_permutation, sabre_result = sabre_layout_and_routing(sabre_dag, neighbor_table, dist_matrix, Heuristic.Decay, self.max_iterations, self.swap_trials, self.layout_trials, self.seed, partial_layouts)
    sabre_stop = time.perf_counter()
    logger.debug('Sabre layout algorithm execution for a connected component complete in: %s sec.', sabre_stop - sabre_start)
    return _DisjointComponent(dag, coupling_map, initial_layout, final_permutation, sabre_result, circuit_to_dag_dict)