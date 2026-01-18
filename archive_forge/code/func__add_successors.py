import heapq
from collections import OrderedDict
from functools import partial
from typing import Sequence, Callable
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.wires import Wires
from pennylane.transforms import transform
def _add_successors(self):
    for node_id in range(len(self._multi_graph) - 1, -1, -1):
        direct_successors = self.direct_successors(node_id)
        for d_succ in direct_successors:
            self.get_node(node_id).successors.append([d_succ])
            self.get_node(node_id).successors.append(self.get_node(d_succ).successors)
        self.get_node(node_id).successors = list(_merge_no_duplicates(*self.get_node(node_id).successors))