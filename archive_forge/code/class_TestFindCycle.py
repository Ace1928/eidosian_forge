from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
class TestFindCycle:

    @classmethod
    def setup_class(cls):
        cls.nodes = [0, 1, 2, 3]
        cls.edges = [(-1, 0), (0, 1), (1, 0), (1, 0), (2, 1), (3, 1)]

    def test_graph_nocycle(self):
        G = nx.Graph(self.edges)
        pytest.raises(nx.exception.NetworkXNoCycle, nx.find_cycle, G, self.nodes)

    def test_graph_cycle(self):
        G = nx.Graph(self.edges)
        G.add_edge(2, 0)
        x = list(nx.find_cycle(G, self.nodes))
        x_ = [(0, 1), (1, 2), (2, 0)]
        assert x == x_

    def test_graph_orientation_none(self):
        G = nx.Graph(self.edges)
        G.add_edge(2, 0)
        x = list(nx.find_cycle(G, self.nodes, orientation=None))
        x_ = [(0, 1), (1, 2), (2, 0)]
        assert x == x_

    def test_graph_orientation_original(self):
        G = nx.Graph(self.edges)
        G.add_edge(2, 0)
        x = list(nx.find_cycle(G, self.nodes, orientation='original'))
        x_ = [(0, 1, FORWARD), (1, 2, FORWARD), (2, 0, FORWARD)]
        assert x == x_

    def test_digraph(self):
        G = nx.DiGraph(self.edges)
        x = list(nx.find_cycle(G, self.nodes))
        x_ = [(0, 1), (1, 0)]
        assert x == x_

    def test_digraph_orientation_none(self):
        G = nx.DiGraph(self.edges)
        x = list(nx.find_cycle(G, self.nodes, orientation=None))
        x_ = [(0, 1), (1, 0)]
        assert x == x_

    def test_digraph_orientation_original(self):
        G = nx.DiGraph(self.edges)
        x = list(nx.find_cycle(G, self.nodes, orientation='original'))
        x_ = [(0, 1, FORWARD), (1, 0, FORWARD)]
        assert x == x_

    def test_multigraph(self):
        G = nx.MultiGraph(self.edges)
        x = list(nx.find_cycle(G, self.nodes))
        x_ = [(0, 1, 0), (1, 0, 1)]
        assert x[0] == x_[0]
        assert x[1][:2] == x_[1][:2]

    def test_multidigraph(self):
        G = nx.MultiDiGraph(self.edges)
        x = list(nx.find_cycle(G, self.nodes))
        x_ = [(0, 1, 0), (1, 0, 0)]
        assert x[0] == x_[0]
        assert x[1][:2] == x_[1][:2]

    def test_digraph_ignore(self):
        G = nx.DiGraph(self.edges)
        x = list(nx.find_cycle(G, self.nodes, orientation='ignore'))
        x_ = [(0, 1, FORWARD), (1, 0, FORWARD)]
        assert x == x_

    def test_digraph_reverse(self):
        G = nx.DiGraph(self.edges)
        x = list(nx.find_cycle(G, self.nodes, orientation='reverse'))
        x_ = [(1, 0, REVERSE), (0, 1, REVERSE)]
        assert x == x_

    def test_multidigraph_ignore(self):
        G = nx.MultiDiGraph(self.edges)
        x = list(nx.find_cycle(G, self.nodes, orientation='ignore'))
        x_ = [(0, 1, 0, FORWARD), (1, 0, 0, FORWARD)]
        assert x[0] == x_[0]
        assert x[1][:2] == x_[1][:2]
        assert x[1][3] == x_[1][3]

    def test_multidigraph_ignore2(self):
        G = nx.MultiDiGraph([(0, 1), (1, 2), (1, 2)])
        x = list(nx.find_cycle(G, [0, 1, 2], orientation='ignore'))
        x_ = [(1, 2, 0, FORWARD), (1, 2, 1, REVERSE)]
        assert x == x_

    def test_multidigraph_original(self):
        G = nx.MultiDiGraph([(0, 1), (1, 2), (2, 3), (4, 2)])
        pytest.raises(nx.exception.NetworkXNoCycle, nx.find_cycle, G, [0, 1, 2, 3, 4], orientation='original')

    def test_dag(self):
        G = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
        pytest.raises(nx.exception.NetworkXNoCycle, nx.find_cycle, G, orientation='original')
        x = list(nx.find_cycle(G, orientation='ignore'))
        assert x == [(0, 1, FORWARD), (1, 2, FORWARD), (0, 2, REVERSE)]

    def test_prev_explored(self):
        G = nx.DiGraph()
        G.add_edges_from([(1, 0), (2, 0), (1, 2), (2, 1)])
        pytest.raises(nx.NetworkXNoCycle, nx.find_cycle, G, source=0)
        x = list(nx.find_cycle(G, 1))
        x_ = [(1, 2), (2, 1)]
        assert x == x_
        x = list(nx.find_cycle(G, 2))
        x_ = [(2, 1), (1, 2)]
        assert x == x_
        x = list(nx.find_cycle(G))
        x_ = [(1, 2), (2, 1)]
        assert x == x_

    def test_no_cycle(self):
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 0), (3, 1), (3, 2)])
        pytest.raises(nx.NetworkXNoCycle, nx.find_cycle, G, source=0)
        pytest.raises(nx.NetworkXNoCycle, nx.find_cycle, G)