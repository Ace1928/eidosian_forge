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
def _pred_update(self, node_id):
    self.get_node(node_id).predecessors = []
    for d_pred in self.direct_predecessors(node_id):
        self.get_node(node_id).predecessors.append([d_pred])
        self.get_node(node_id).predecessors.append(self.get_node(d_pred).predecessors)
    self.get_node(node_id).predecessors = list(_merge_no_duplicates(*self.get_node(node_id).predecessors))