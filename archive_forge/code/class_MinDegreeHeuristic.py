import itertools
import sys
from heapq import heapify, heappop, heappush
import networkx as nx
from networkx.utils import not_implemented_for
class MinDegreeHeuristic:
    """Implements the Minimum Degree heuristic.

    The heuristic chooses the nodes according to their degree
    (number of neighbours), i.e., first the node with the lowest degree is
    chosen, then the graph is updated and the corresponding node is
    removed. Next, a new node with the lowest degree is chosen, and so on.
    """

    def __init__(self, graph):
        self._graph = graph
        self._update_nodes = []
        self._degreeq = []
        self.count = itertools.count()
        for n in graph:
            self._degreeq.append((len(graph[n]), next(self.count), n))
        heapify(self._degreeq)

    def best_node(self, graph):
        for n in self._update_nodes:
            heappush(self._degreeq, (len(graph[n]), next(self.count), n))
        while self._degreeq:
            min_degree, _, elim_node = heappop(self._degreeq)
            if elim_node not in graph or len(graph[elim_node]) != min_degree:
                continue
            elif min_degree == len(graph) - 1:
                return None
            self._update_nodes = graph[elim_node]
            return elim_node
        return None