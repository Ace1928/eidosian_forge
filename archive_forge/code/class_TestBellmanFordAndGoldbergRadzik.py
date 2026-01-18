import pytest
import networkx as nx
from networkx.utils import pairwise
class TestBellmanFordAndGoldbergRadzik(WeightedTestBase):

    def test_single_node_graph(self):
        G = nx.DiGraph()
        G.add_node(0)
        assert nx.single_source_bellman_ford_path(G, 0) == {0: [0]}
        assert nx.single_source_bellman_ford_path_length(G, 0) == {0: 0}
        assert nx.single_source_bellman_ford(G, 0) == ({0: 0}, {0: [0]})
        assert nx.bellman_ford_predecessor_and_distance(G, 0) == ({0: []}, {0: 0})
        assert nx.goldberg_radzik(G, 0) == ({0: None}, {0: 0})

    def test_absent_source_bellman_ford(self):
        G = nx.path_graph(2)
        for fn in (nx.bellman_ford_predecessor_and_distance, nx.bellman_ford_path, nx.bellman_ford_path_length, nx.single_source_bellman_ford_path, nx.single_source_bellman_ford_path_length, nx.single_source_bellman_ford):
            pytest.raises(nx.NodeNotFound, fn, G, 3, 0)
            pytest.raises(nx.NodeNotFound, fn, G, 3, 3)

    def test_absent_source_goldberg_radzik(self):
        with pytest.raises(nx.NodeNotFound):
            G = nx.path_graph(2)
            nx.goldberg_radzik(G, 3, 0)

    def test_negative_cycle_heuristic(self):
        G = nx.DiGraph()
        G.add_edge(0, 1, weight=-1)
        G.add_edge(1, 2, weight=-1)
        G.add_edge(2, 3, weight=-1)
        G.add_edge(3, 0, weight=3)
        assert not nx.negative_edge_cycle(G, heuristic=True)
        G.add_edge(2, 0, weight=1.999)
        assert nx.negative_edge_cycle(G, heuristic=True)
        G.edges[2, 0]['weight'] = 2
        assert not nx.negative_edge_cycle(G, heuristic=True)

    def test_negative_cycle_consistency(self):
        import random
        unif = random.uniform
        for random_seed in range(2):
            random.seed(random_seed)
            for density in [0.1, 0.9]:
                for N in [1, 10, 20]:
                    for max_cost in [1, 90]:
                        G = nx.binomial_graph(N, density, seed=4, directed=True)
                        edges = ((u, v, unif(-1, max_cost)) for u, v in G.edges)
                        G.add_weighted_edges_from(edges)
                        no_heuristic = nx.negative_edge_cycle(G, heuristic=False)
                        with_heuristic = nx.negative_edge_cycle(G, heuristic=True)
                        assert no_heuristic == with_heuristic

    def test_negative_cycle(self):
        G = nx.cycle_graph(5, create_using=nx.DiGraph())
        G.add_edge(1, 2, weight=-7)
        for i in range(5):
            pytest.raises(nx.NetworkXUnbounded, nx.single_source_bellman_ford_path, G, i)
            pytest.raises(nx.NetworkXUnbounded, nx.single_source_bellman_ford_path_length, G, i)
            pytest.raises(nx.NetworkXUnbounded, nx.single_source_bellman_ford, G, i)
            pytest.raises(nx.NetworkXUnbounded, nx.bellman_ford_predecessor_and_distance, G, i)
            pytest.raises(nx.NetworkXUnbounded, nx.goldberg_radzik, G, i)
        G = nx.cycle_graph(5)
        G.add_edge(1, 2, weight=-3)
        for i in range(5):
            pytest.raises(nx.NetworkXUnbounded, nx.single_source_bellman_ford_path, G, i)
            pytest.raises(nx.NetworkXUnbounded, nx.single_source_bellman_ford_path_length, G, i)
            pytest.raises(nx.NetworkXUnbounded, nx.single_source_bellman_ford, G, i)
            pytest.raises(nx.NetworkXUnbounded, nx.bellman_ford_predecessor_and_distance, G, i)
            pytest.raises(nx.NetworkXUnbounded, nx.goldberg_radzik, G, i)
        G = nx.DiGraph([(1, 1, {'weight': -1})])
        pytest.raises(nx.NetworkXUnbounded, nx.single_source_bellman_ford_path, G, 1)
        pytest.raises(nx.NetworkXUnbounded, nx.single_source_bellman_ford_path_length, G, 1)
        pytest.raises(nx.NetworkXUnbounded, nx.single_source_bellman_ford, G, 1)
        pytest.raises(nx.NetworkXUnbounded, nx.bellman_ford_predecessor_and_distance, G, 1)
        pytest.raises(nx.NetworkXUnbounded, nx.goldberg_radzik, G, 1)
        G = nx.MultiDiGraph([(1, 1, {'weight': -1})])
        pytest.raises(nx.NetworkXUnbounded, nx.single_source_bellman_ford_path, G, 1)
        pytest.raises(nx.NetworkXUnbounded, nx.single_source_bellman_ford_path_length, G, 1)
        pytest.raises(nx.NetworkXUnbounded, nx.single_source_bellman_ford, G, 1)
        pytest.raises(nx.NetworkXUnbounded, nx.bellman_ford_predecessor_and_distance, G, 1)
        pytest.raises(nx.NetworkXUnbounded, nx.goldberg_radzik, G, 1)

    def test_zero_cycle(self):
        G = nx.cycle_graph(5, create_using=nx.DiGraph())
        G.add_edge(2, 3, weight=-4)
        nx.goldberg_radzik(G, 1)
        nx.bellman_ford_predecessor_and_distance(G, 1)
        G.add_edge(2, 3, weight=-4.0001)
        pytest.raises(nx.NetworkXUnbounded, nx.bellman_ford_predecessor_and_distance, G, 1)
        pytest.raises(nx.NetworkXUnbounded, nx.goldberg_radzik, G, 1)

    def test_find_negative_cycle_longer_cycle(self):
        G = nx.cycle_graph(5, create_using=nx.DiGraph())
        nx.add_cycle(G, [3, 5, 6, 7, 8, 9])
        G.add_edge(1, 2, weight=-30)
        assert nx.find_negative_cycle(G, 1) == [0, 1, 2, 3, 4, 0]
        assert nx.find_negative_cycle(G, 7) == [2, 3, 4, 0, 1, 2]

    def test_find_negative_cycle_no_cycle(self):
        G = nx.path_graph(5, create_using=nx.DiGraph())
        pytest.raises(nx.NetworkXError, nx.find_negative_cycle, G, 3)

    def test_find_negative_cycle_single_edge(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=-1)
        assert nx.find_negative_cycle(G, 1) == [1, 0, 1]

    def test_negative_weight(self):
        G = nx.cycle_graph(5, create_using=nx.DiGraph())
        G.add_edge(1, 2, weight=-3)
        assert nx.single_source_bellman_ford_path(G, 0) == {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 2, 3], 4: [0, 1, 2, 3, 4]}
        assert nx.single_source_bellman_ford_path_length(G, 0) == {0: 0, 1: 1, 2: -2, 3: -1, 4: 0}
        assert nx.single_source_bellman_ford(G, 0) == ({0: 0, 1: 1, 2: -2, 3: -1, 4: 0}, {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 2, 3], 4: [0, 1, 2, 3, 4]})
        assert nx.bellman_ford_predecessor_and_distance(G, 0) == ({0: [], 1: [0], 2: [1], 3: [2], 4: [3]}, {0: 0, 1: 1, 2: -2, 3: -1, 4: 0})
        assert nx.goldberg_radzik(G, 0) == ({0: None, 1: 0, 2: 1, 3: 2, 4: 3}, {0: 0, 1: 1, 2: -2, 3: -1, 4: 0})

    def test_not_connected(self):
        G = nx.complete_graph(6)
        G.add_edge(10, 11)
        G.add_edge(10, 12)
        assert nx.single_source_bellman_ford_path(G, 0) == {0: [0], 1: [0, 1], 2: [0, 2], 3: [0, 3], 4: [0, 4], 5: [0, 5]}
        assert nx.single_source_bellman_ford_path_length(G, 0) == {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
        assert nx.single_source_bellman_ford(G, 0) == ({0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}, {0: [0], 1: [0, 1], 2: [0, 2], 3: [0, 3], 4: [0, 4], 5: [0, 5]})
        assert nx.bellman_ford_predecessor_and_distance(G, 0) == ({0: [], 1: [0], 2: [0], 3: [0], 4: [0], 5: [0]}, {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
        assert nx.goldberg_radzik(G, 0) == ({0: None, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
        G = nx.complete_graph(6)
        G.add_edges_from([('A', 'B', {'load': 3}), ('B', 'C', {'load': -10}), ('C', 'A', {'load': 2})])
        assert nx.single_source_bellman_ford_path(G, 0, weight='load') == {0: [0], 1: [0, 1], 2: [0, 2], 3: [0, 3], 4: [0, 4], 5: [0, 5]}
        assert nx.single_source_bellman_ford_path_length(G, 0, weight='load') == {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
        assert nx.single_source_bellman_ford(G, 0, weight='load') == ({0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}, {0: [0], 1: [0, 1], 2: [0, 2], 3: [0, 3], 4: [0, 4], 5: [0, 5]})
        assert nx.bellman_ford_predecessor_and_distance(G, 0, weight='load') == ({0: [], 1: [0], 2: [0], 3: [0], 4: [0], 5: [0]}, {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
        assert nx.goldberg_radzik(G, 0, weight='load') == ({0: None, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1})

    def test_multigraph(self):
        assert nx.bellman_ford_path(self.MXG, 's', 'v') == ['s', 'x', 'u', 'v']
        assert nx.bellman_ford_path_length(self.MXG, 's', 'v') == 9
        assert nx.single_source_bellman_ford_path(self.MXG, 's')['v'] == ['s', 'x', 'u', 'v']
        assert nx.single_source_bellman_ford_path_length(self.MXG, 's')['v'] == 9
        D, P = nx.single_source_bellman_ford(self.MXG, 's', target='v')
        assert D == 9
        assert P == ['s', 'x', 'u', 'v']
        P, D = nx.bellman_ford_predecessor_and_distance(self.MXG, 's')
        assert P['v'] == ['u']
        assert D['v'] == 9
        P, D = nx.goldberg_radzik(self.MXG, 's')
        assert P['v'] == 'u'
        assert D['v'] == 9
        assert nx.bellman_ford_path(self.MXG4, 0, 2) == [0, 1, 2]
        assert nx.bellman_ford_path_length(self.MXG4, 0, 2) == 4
        assert nx.single_source_bellman_ford_path(self.MXG4, 0)[2] == [0, 1, 2]
        assert nx.single_source_bellman_ford_path_length(self.MXG4, 0)[2] == 4
        D, P = nx.single_source_bellman_ford(self.MXG4, 0, target=2)
        assert D == 4
        assert P == [0, 1, 2]
        P, D = nx.bellman_ford_predecessor_and_distance(self.MXG4, 0)
        assert P[2] == [1]
        assert D[2] == 4
        P, D = nx.goldberg_radzik(self.MXG4, 0)
        assert P[2] == 1
        assert D[2] == 4

    def test_others(self):
        assert nx.bellman_ford_path(self.XG, 's', 'v') == ['s', 'x', 'u', 'v']
        assert nx.bellman_ford_path_length(self.XG, 's', 'v') == 9
        assert nx.single_source_bellman_ford_path(self.XG, 's')['v'] == ['s', 'x', 'u', 'v']
        assert nx.single_source_bellman_ford_path_length(self.XG, 's')['v'] == 9
        D, P = nx.single_source_bellman_ford(self.XG, 's', target='v')
        assert D == 9
        assert P == ['s', 'x', 'u', 'v']
        P, D = nx.bellman_ford_predecessor_and_distance(self.XG, 's')
        assert P['v'] == ['u']
        assert D['v'] == 9
        P, D = nx.goldberg_radzik(self.XG, 's')
        assert P['v'] == 'u'
        assert D['v'] == 9

    def test_path_graph(self):
        G = nx.path_graph(4)
        assert nx.single_source_bellman_ford_path(G, 0) == {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 2, 3]}
        assert nx.single_source_bellman_ford_path_length(G, 0) == {0: 0, 1: 1, 2: 2, 3: 3}
        assert nx.single_source_bellman_ford(G, 0) == ({0: 0, 1: 1, 2: 2, 3: 3}, {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 2, 3]})
        assert nx.bellman_ford_predecessor_and_distance(G, 0) == ({0: [], 1: [0], 2: [1], 3: [2]}, {0: 0, 1: 1, 2: 2, 3: 3})
        assert nx.goldberg_radzik(G, 0) == ({0: None, 1: 0, 2: 1, 3: 2}, {0: 0, 1: 1, 2: 2, 3: 3})
        assert nx.single_source_bellman_ford_path(G, 3) == {0: [3, 2, 1, 0], 1: [3, 2, 1], 2: [3, 2], 3: [3]}
        assert nx.single_source_bellman_ford_path_length(G, 3) == {0: 3, 1: 2, 2: 1, 3: 0}
        assert nx.single_source_bellman_ford(G, 3) == ({0: 3, 1: 2, 2: 1, 3: 0}, {0: [3, 2, 1, 0], 1: [3, 2, 1], 2: [3, 2], 3: [3]})
        assert nx.bellman_ford_predecessor_and_distance(G, 3) == ({0: [1], 1: [2], 2: [3], 3: []}, {0: 3, 1: 2, 2: 1, 3: 0})
        assert nx.goldberg_radzik(G, 3) == ({0: 1, 1: 2, 2: 3, 3: None}, {0: 3, 1: 2, 2: 1, 3: 0})

    def test_4_cycle(self):
        G = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0)])
        dist, path = nx.single_source_bellman_ford(G, 0)
        assert dist == {0: 0, 1: 1, 2: 2, 3: 1}
        assert path[0] == [0]
        assert path[1] == [0, 1]
        assert path[2] in [[0, 1, 2], [0, 3, 2]]
        assert path[3] == [0, 3]
        pred, dist = nx.bellman_ford_predecessor_and_distance(G, 0)
        assert pred[0] == []
        assert pred[1] == [0]
        assert pred[2] in [[1, 3], [3, 1]]
        assert pred[3] == [0]
        assert dist == {0: 0, 1: 1, 2: 2, 3: 1}
        pred, dist = nx.goldberg_radzik(G, 0)
        assert pred[0] is None
        assert pred[1] == 0
        assert pred[2] in [1, 3]
        assert pred[3] == 0
        assert dist == {0: 0, 1: 1, 2: 2, 3: 1}

    def test_negative_weight_bf_path(self):
        G = nx.DiGraph()
        G.add_nodes_from('abcd')
        G.add_edge('a', 'd', weight=0)
        G.add_edge('a', 'b', weight=1)
        G.add_edge('b', 'c', weight=-3)
        G.add_edge('c', 'd', weight=1)
        assert nx.bellman_ford_path(G, 'a', 'd') == ['a', 'b', 'c', 'd']
        assert nx.bellman_ford_path_length(G, 'a', 'd') == -1

    def test_zero_cycle_smoke(self):
        D = nx.DiGraph()
        D.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 1, -2)])
        nx.bellman_ford_path(D, 1, 3)
        nx.dijkstra_path(D, 1, 3)
        nx.bidirectional_dijkstra(D, 1, 3)