import heapq
import math
from itertools import chain, combinations, zip_longest
from operator import itemgetter
import networkx as nx
from networkx.utils import py_random_state, random_weighted_sample
class DegreeSequenceRandomGraph:

    def __init__(self, degree, rng):
        if not nx.is_graphical(degree):
            raise nx.NetworkXUnfeasible('degree sequence is not graphical')
        self.rng = rng
        self.degree = list(degree)
        self.m = sum(self.degree) / 2.0
        try:
            self.dmax = max(self.degree)
        except ValueError:
            self.dmax = 0

    def generate(self):
        self.remaining_degree = dict(enumerate(self.degree))
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.remaining_degree)
        for n, d in list(self.remaining_degree.items()):
            if d == 0:
                del self.remaining_degree[n]
        if len(self.remaining_degree) > 0:
            self.phase1()
            self.phase2()
            self.phase3()
        return self.graph

    def update_remaining(self, u, v, aux_graph=None):
        if aux_graph is not None:
            aux_graph.remove_edge(u, v)
        if self.remaining_degree[u] == 1:
            del self.remaining_degree[u]
            if aux_graph is not None:
                aux_graph.remove_node(u)
        else:
            self.remaining_degree[u] -= 1
        if self.remaining_degree[v] == 1:
            del self.remaining_degree[v]
            if aux_graph is not None:
                aux_graph.remove_node(v)
        else:
            self.remaining_degree[v] -= 1

    def p(self, u, v):
        return 1 - self.degree[u] * self.degree[v] / (4.0 * self.m)

    def q(self, u, v):
        norm = max(self.remaining_degree.values()) ** 2
        return self.remaining_degree[u] * self.remaining_degree[v] / norm

    def suitable_edge(self):
        """Returns True if and only if an arbitrary remaining node can
        potentially be joined with some other remaining node.

        """
        nodes = iter(self.remaining_degree)
        u = next(nodes)
        return any((v not in self.graph[u] for v in nodes))

    def phase1(self):
        rem_deg = self.remaining_degree
        while sum(rem_deg.values()) >= 2 * self.dmax ** 2:
            u, v = sorted(random_weighted_sample(rem_deg, 2, self.rng))
            if self.graph.has_edge(u, v):
                continue
            if self.rng.random() < self.p(u, v):
                self.graph.add_edge(u, v)
                self.update_remaining(u, v)

    def phase2(self):
        remaining_deg = self.remaining_degree
        rng = self.rng
        while len(remaining_deg) >= 2 * self.dmax:
            while True:
                u, v = sorted(rng.sample(list(remaining_deg.keys()), 2))
                if self.graph.has_edge(u, v):
                    continue
                if rng.random() < self.q(u, v):
                    break
            if rng.random() < self.p(u, v):
                self.graph.add_edge(u, v)
                self.update_remaining(u, v)

    def phase3(self):
        potential_edges = combinations(self.remaining_degree, 2)
        H = nx.Graph([(u, v) for u, v in potential_edges if not self.graph.has_edge(u, v)])
        rng = self.rng
        while self.remaining_degree:
            if not self.suitable_edge():
                raise nx.NetworkXUnfeasible('no suitable edges left')
            while True:
                u, v = sorted(rng.choice(list(H.edges())))
                if rng.random() < self.q(u, v):
                    break
            if rng.random() < self.p(u, v):
                self.graph.add_edge(u, v)
                self.update_remaining(u, v, aux_graph=H)