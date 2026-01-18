import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
def _clear_partition(self, G):
    """
        Removes partition data from the graph
        """
    for u, v, d in G.edges(data=True):
        if self.partition_key in d:
            del d[self.partition_key]