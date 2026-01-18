from numbers import Number
from collections import namedtuple
import numpy as np
import rustworkx as rx
from pennylane.measurements import MeasurementProcess
from pennylane.resource import ResourcesOperation
def _in_topological_order(self, ops):
    """Sorts a set of operators in the circuit in a topological order.

        Args:
            ops (Iterable[Operator]): set of operators in the circuit

        Returns:
            Iterable[Operator]: same set of operators, topologically ordered
        """
    G = self._graph.subgraph(list((self._indices[id(o)] for o in ops)))
    indexes = rx.topological_sort(G)
    return list((G[x] for x in indexes))