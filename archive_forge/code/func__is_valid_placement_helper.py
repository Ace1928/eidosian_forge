import abc
import warnings
from dataclasses import dataclass
from typing import (
import networkx as nx
from matplotlib import pyplot as plt
from cirq import _compat
from cirq.devices import GridQubit, LineQubit
from cirq.protocols.json_serialization import dataclass_json_dict
def _is_valid_placement_helper(big_graph: nx.Graph, small_mapped: nx.Graph, small_to_big_mapping: Dict):
    """Helper function for `is_valid_placement` that assumes the mapping of `small_graph` has
    already occurred.

    This is so we don't duplicate work when checking placements during `draw_placements`.
    """
    subgraph = big_graph.subgraph(small_to_big_mapping.values())
    return subgraph.nodes == small_mapped.nodes and subgraph.edges == small_mapped.edges