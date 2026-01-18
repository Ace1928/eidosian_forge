import abc
import warnings
from dataclasses import dataclass
from typing import (
import networkx as nx
from matplotlib import pyplot as plt
from cirq import _compat
from cirq.devices import GridQubit, LineQubit
from cirq.protocols.json_serialization import dataclass_json_dict
@dataclass(frozen=True)
class LineTopology(NamedTopology):
    """A 1D linear topology.

    Node indices are contiguous integers starting from 0 with edges between
    adjacent integers.

    Args:
        n_nodes: The number of nodes in a line.
    """
    n_nodes: int

    def __post_init__(self):
        if self.n_nodes <= 1:
            raise ValueError('`n_nodes` must be greater than 1.')
        object.__setattr__(self, 'name', f'line-{self.n_nodes}')
        graph = nx.from_edgelist([(i1, i2) for i1, i2 in zip(range(self.n_nodes), range(1, self.n_nodes))])
        object.__setattr__(self, 'graph', graph)

    def nodes_as_linequbits(self) -> List['cirq.LineQubit']:
        """Get the graph nodes as cirq.LineQubit"""
        return [LineQubit(x) for x in sorted(self.graph.nodes)]

    def draw(self, ax=None, tilted: bool=True, **kwargs) -> Dict[Any, Tuple[int, int]]:
        """Draw this graph using Matplotlib.

        Args:
            ax: Optional matplotlib axis to use for drawing.
            tilted: If True, draw as a horizontal line. Otherwise, draw on a diagonal.
            **kwargs: Additional arguments to pass to `nx.draw_networkx`.
        """
        g2 = nx.relabel_nodes(self.graph, {n: (n, 1) for n in self.graph.nodes})
        return draw_gridlike(g2, ax=ax, tilted=tilted, **kwargs)

    def nodes_to_linequbits(self, offset: int=0) -> Dict[int, 'cirq.LineQubit']:
        """Return a mapping from graph nodes to `cirq.LineQubit`

        Args:
            offset: Offset integer positions of the resultant LineQubits by this amount.
        """
        return dict(enumerate(LineQubit.range(offset, offset + self.n_nodes)))

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self)

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self)