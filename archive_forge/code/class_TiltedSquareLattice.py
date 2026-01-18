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
class TiltedSquareLattice(NamedTopology):
    """A grid lattice rotated 45-degrees.

    This topology is based on Google devices where plaquettes consist of four qubits in a square
    connected to a central qubit:

        x   x
          x
        x   x

    The corner nodes are not connected to each other. `width` and `height` refer to the rectangle
    formed by rotating the lattice 45 degrees. `width` and `height` are measured in half-unit
    cells, or equivalently half the number of central nodes.
    An example diagram of this topology is shown below. It is a
    "tilted-square-lattice-6-4" with width 6 and height 4.

              x
              │
         x────X────x
         │    │    │
    x────X────x────X────x
         │    │    │    │
         x────X────x────X───x
              │    │    │
              x────X────x
                   │
                   x

    Nodes are 2-tuples of integers which may be negative. Please see `get_placements` for
    mapping this topology to a GridQubit Device.
    """
    width: int
    height: int

    def __post_init__(self):
        if self.width <= 0:
            raise ValueError('Width must be a positive integer')
        if self.height <= 0:
            raise ValueError('Height must be a positive integer')
        object.__setattr__(self, 'name', f'tilted-square-lattice-{self.width}-{self.height}')
        rect1 = set(((i + j, i - j) for i in range(self.width // 2 + 1) for j in range(self.height // 2 + 1)))
        rect2 = set((((i + j) // 2, (i - j) // 2) for i in range(1, self.width + 1, 2) for j in range(1, self.height + 1, 2)))
        nodes = rect1 | rect2
        g = nx.Graph()
        for node in nodes:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (node[0] + dx, node[1] + dy)
                if neighbor in nodes:
                    g.add_edge(node, neighbor)
        object.__setattr__(self, 'graph', g)
        n_nodes = (self.width // 2 + 1) * (self.height // 2 + 1)
        n_nodes += (self.width + 1) // 2 * ((self.height + 1) // 2)
        object.__setattr__(self, 'n_nodes', n_nodes)

    def draw(self, ax=None, tilted=True, **kwargs):
        """Draw this graph using Matplotlib.

        Args:
            ax: Optional matplotlib axis to use for drawing.
            tilted: If True, directly position as (row, column); otherwise,
                rotate 45 degrees to accommodate the diagonal nature of this topology.
            **kwargs: Additional arguments to pass to `nx.draw_networkx`.
        """
        return draw_gridlike(self.graph, ax=ax, tilted=tilted, **kwargs)

    def nodes_as_gridqubits(self) -> List['cirq.GridQubit']:
        """Get the graph nodes as cirq.GridQubit"""
        return [GridQubit(r, c) for r, c in sorted(self.graph.nodes)]

    def nodes_to_gridqubits(self, offset=(0, 0)) -> Dict[Tuple[int, int], 'cirq.GridQubit']:
        """Return a mapping from graph nodes to `cirq.GridQubit`

        Args:
            offset: Offset row and column indices of the resultant GridQubits by this amount.
                The offset positions the top-left node in the `draw(tilted=False)` frame.
        """
        return {(r, c): GridQubit(r, c) + offset for r, c in self.graph.nodes}

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self)

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self)