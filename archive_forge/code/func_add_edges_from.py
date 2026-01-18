import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
def add_edges_from(self, ebunch_to_add, **attr):
    for u, v, k, d in ebunch_to_add:
        self.add_edge(u, v, k, **d)