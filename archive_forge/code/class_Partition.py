import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
@dataclass(order=True)
class Partition:
    """
        This dataclass represents a partition and stores a dict with the edge
        data and the weight of the minimum spanning arborescence of the
        partition dict.
        """
    mst_weight: float
    partition_dict: dict = field(compare=False)

    def __copy__(self):
        return ArborescenceIterator.Partition(self.mst_weight, self.partition_dict.copy())