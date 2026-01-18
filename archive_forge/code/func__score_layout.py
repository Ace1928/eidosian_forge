from enum import Enum
import logging
import inspect
import itertools
import time
from rustworkx import PyDiGraph, vf2_mapping, PyGraph
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.transpiler.passes.layout import vf2_utils
def _score_layout(self, layout, bit_map, reverse_bit_map, im_graph):
    bits = layout.get_virtual_bits()
    fidelity = 1
    if self.target is not None:
        for bit, node_index in bit_map.items():
            gate_counts = im_graph[node_index]
            for gate, count in gate_counts.items():
                if self.target[gate] is not None and None not in self.target[gate]:
                    props = self.target[gate][bits[bit],]
                    if props is not None and props.error is not None:
                        fidelity *= (1 - props.error) ** count
        for edge in im_graph.edge_index_map().values():
            qargs = (bits[reverse_bit_map[edge[0]]], bits[reverse_bit_map[edge[1]]])
            gate_counts = edge[2]
            for gate, count in gate_counts.items():
                if self.target[gate] is not None and None not in self.target[gate]:
                    props = self.target[gate][qargs]
                    if props is not None and props.error is not None:
                        fidelity *= (1 - props.error) ** count
    else:
        for bit, node_index in bit_map.items():
            gate_counts = im_graph[node_index]
            for gate, count in gate_counts.items():
                if gate == 'measure':
                    try:
                        fidelity *= (1 - self.properties.readout_error(bits[bit])) ** count
                    except BackendPropertyError:
                        pass
                else:
                    try:
                        fidelity *= (1 - self.properties.gate_error(gate, bits[bit])) ** count
                    except BackendPropertyError:
                        pass
        for edge in im_graph.edge_index_map().values():
            qargs = (bits[reverse_bit_map[edge[0]]], bits[reverse_bit_map[edge[1]]])
            gate_counts = edge[2]
            for gate, count in gate_counts.items():
                try:
                    fidelity *= (1 - self.properties.gate_error(gate, qargs)) ** count
                except BackendPropertyError:
                    pass
    return 1 - fidelity