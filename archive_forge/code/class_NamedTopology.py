import abc
import warnings
from dataclasses import dataclass
from typing import (
import networkx as nx
from matplotlib import pyplot as plt
from cirq import _compat
from cirq.devices import GridQubit, LineQubit
from cirq.protocols.json_serialization import dataclass_json_dict
class NamedTopology(metaclass=abc.ABCMeta):
    """A topology (graph) with a name.

    "Named topologies" provide a mapping from a simple dataclass to a unique graph for categories
    of relevant topologies. Relevant topologies may be hardware dependant, but common topologies
    are linear (1D) and rectangular grid topologies.
    """
    name: str = NotImplemented
    'A name that uniquely identifies this topology.'
    n_nodes: int = NotImplemented
    'The number of nodes in the topology.'
    graph: nx.Graph = NotImplemented
    'A networkx graph representation of the topology.'