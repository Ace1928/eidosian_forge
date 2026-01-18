import pytest
import networkx as nx
class TestGenericPath:

    @classmethod
    def setup_class(cls):
        from networkx import convert_node_labels_to_integers as cnlti
        cls.grid = cnlti(nx.grid_2d_graph(4, 4), first_label=1, ordering='sorted')
        cls.cycle = nx.cycle_graph(7)
        cls.directed_cycle = nx.cycle_graph(7, create_using=nx.DiGraph())
        cls.neg_weights = nx.DiGraph()
        cls.neg_weights.add_edge(0, 1, weight=1)
        cls.neg_weights.add_edge(0, 2, weight=3)
        cls.neg_weights.add_edge(1, 3, weight=1)
        cls.neg_weights.add_edge(2, 3, weight=-2)

    def test_shortest_path(self):
        assert nx.shortest_path(self.cycle, 0, 3) == [0, 1, 2, 3]
        assert nx.shortest_path(self.cycle, 0, 4) == [0, 6, 5, 4]
        validate_grid_path(4, 4, 1, 12, nx.shortest_path(self.grid, 1, 12))
        assert nx.shortest_path(self.directed_cycle, 0, 3) == [0, 1, 2, 3]
        assert nx.shortest_path(self.cycle, 0, 3, weight='weight') == [0, 1, 2, 3]
        assert nx.shortest_path(self.cycle, 0, 4, weight='weight') == [0, 6, 5, 4]
        validate_grid_path(4, 4, 1, 12, nx.shortest_path(self.grid, 1, 12, weight='weight'))
        assert nx.shortest_path(self.directed_cycle, 0, 3, weight='weight') == [0, 1, 2, 3]
        assert nx.shortest_path(self.directed_cycle, 0, 3, weight='weight', method='dijkstra') == [0, 1, 2, 3]
        assert nx.shortest_path(self.directed_cycle, 0, 3, weight='weight', method='bellman-ford') == [0, 1, 2, 3]
        assert nx.shortest_path(self.neg_weights, 0, 3, weight='weight', method='bellman-ford') == [0, 2, 3]
        pytest.raises(ValueError, nx.shortest_path, self.cycle, method='SPAM')
        pytest.raises(nx.NodeNotFound, nx.shortest_path, self.cycle, 8)

    def test_shortest_path_target(self):
        answer = {0: [0, 1], 1: [1], 2: [2, 1]}
        sp = nx.shortest_path(nx.path_graph(3), target=1)
        assert sp == answer
        sp = nx.shortest_path(nx.path_graph(3), target=1, weight='weight')
        assert sp == answer
        sp = nx.shortest_path(nx.path_graph(3), target=1, weight='weight', method='dijkstra')
        assert sp == answer
        sp = nx.shortest_path(nx.path_graph(3), target=1, weight='weight', method='bellman-ford')
        assert sp == answer

    def test_shortest_path_length(self):
        assert nx.shortest_path_length(self.cycle, 0, 3) == 3
        assert nx.shortest_path_length(self.grid, 1, 12) == 5
        assert nx.shortest_path_length(self.directed_cycle, 0, 4) == 4
        assert nx.shortest_path_length(self.cycle, 0, 3, weight='weight') == 3
        assert nx.shortest_path_length(self.grid, 1, 12, weight='weight') == 5
        assert nx.shortest_path_length(self.directed_cycle, 0, 4, weight='weight') == 4
        assert nx.shortest_path_length(self.cycle, 0, 3, weight='weight', method='dijkstra') == 3
        assert nx.shortest_path_length(self.cycle, 0, 3, weight='weight', method='bellman-ford') == 3
        pytest.raises(ValueError, nx.shortest_path_length, self.cycle, method='SPAM')
        pytest.raises(nx.NodeNotFound, nx.shortest_path_length, self.cycle, 8)

    def test_shortest_path_length_target(self):
        answer = {0: 1, 1: 0, 2: 1}
        sp = dict(nx.shortest_path_length(nx.path_graph(3), target=1))
        assert sp == answer
        sp = nx.shortest_path_length(nx.path_graph(3), target=1, weight='weight')
        assert sp == answer
        sp = nx.shortest_path_length(nx.path_graph(3), target=1, weight='weight', method='dijkstra')
        assert sp == answer
        sp = nx.shortest_path_length(nx.path_graph(3), target=1, weight='weight', method='bellman-ford')
        assert sp == answer

    def test_single_source_shortest_path(self):
        p = nx.shortest_path(self.cycle, 0)
        assert p[3] == [0, 1, 2, 3]
        assert p == nx.single_source_shortest_path(self.cycle, 0)
        p = nx.shortest_path(self.grid, 1)
        validate_grid_path(4, 4, 1, 12, p[12])
        p = nx.shortest_path(self.cycle, 0, weight='weight')
        assert p[3] == [0, 1, 2, 3]
        assert p == nx.single_source_dijkstra_path(self.cycle, 0)
        p = nx.shortest_path(self.grid, 1, weight='weight')
        validate_grid_path(4, 4, 1, 12, p[12])
        p = nx.shortest_path(self.cycle, 0, method='dijkstra', weight='weight')
        assert p[3] == [0, 1, 2, 3]
        assert p == nx.single_source_shortest_path(self.cycle, 0)
        p = nx.shortest_path(self.cycle, 0, method='bellman-ford', weight='weight')
        assert p[3] == [0, 1, 2, 3]
        assert p == nx.single_source_shortest_path(self.cycle, 0)

    def test_single_source_shortest_path_length(self):
        ans = dict(nx.shortest_path_length(self.cycle, 0))
        assert ans == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
        assert ans == dict(nx.single_source_shortest_path_length(self.cycle, 0))
        ans = dict(nx.shortest_path_length(self.grid, 1))
        assert ans[16] == 6
        ans = dict(nx.shortest_path_length(self.cycle, 0, weight='weight'))
        assert ans == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
        assert ans == dict(nx.single_source_dijkstra_path_length(self.cycle, 0))
        ans = dict(nx.shortest_path_length(self.grid, 1, weight='weight'))
        assert ans[16] == 6
        ans = dict(nx.shortest_path_length(self.cycle, 0, weight='weight', method='dijkstra'))
        assert ans == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
        assert ans == dict(nx.single_source_dijkstra_path_length(self.cycle, 0))
        ans = dict(nx.shortest_path_length(self.cycle, 0, weight='weight', method='bellman-ford'))
        assert ans == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
        assert ans == dict(nx.single_source_bellman_ford_path_length(self.cycle, 0))

    def test_single_source_all_shortest_paths(self):
        cycle_ans = {0: [[0]], 1: [[0, 1]], 2: [[0, 1, 2], [0, 3, 2]], 3: [[0, 3]]}
        ans = dict(nx.single_source_all_shortest_paths(nx.cycle_graph(4), 0))
        assert sorted(ans[2]) == cycle_ans[2]
        ans = dict(nx.single_source_all_shortest_paths(self.grid, 1))
        grid_ans = [[1, 2, 3, 7, 11], [1, 2, 6, 7, 11], [1, 2, 6, 10, 11], [1, 5, 6, 7, 11], [1, 5, 6, 10, 11], [1, 5, 9, 10, 11]]
        assert sorted(ans[11]) == grid_ans
        ans = dict(nx.single_source_all_shortest_paths(nx.cycle_graph(4), 0, weight='weight'))
        assert sorted(ans[2]) == cycle_ans[2]
        ans = dict(nx.single_source_all_shortest_paths(nx.cycle_graph(4), 0, method='bellman-ford', weight='weight'))
        assert sorted(ans[2]) == cycle_ans[2]
        ans = dict(nx.single_source_all_shortest_paths(self.grid, 1, weight='weight'))
        assert sorted(ans[11]) == grid_ans
        ans = dict(nx.single_source_all_shortest_paths(self.grid, 1, method='bellman-ford', weight='weight'))
        assert sorted(ans[11]) == grid_ans
        G = nx.cycle_graph(4)
        G.add_node(4)
        ans = dict(nx.single_source_all_shortest_paths(G, 0))
        assert sorted(ans[2]) == [[0, 1, 2], [0, 3, 2]]
        ans = dict(nx.single_source_all_shortest_paths(G, 4))
        assert sorted(ans[4]) == [[4]]

    def test_all_pairs_shortest_path(self):
        p = nx.shortest_path(self.cycle)
        assert p[0][3] == [0, 1, 2, 3]
        assert p == dict(nx.all_pairs_shortest_path(self.cycle))
        p = nx.shortest_path(self.grid)
        validate_grid_path(4, 4, 1, 12, p[1][12])
        p = nx.shortest_path(self.cycle, weight='weight')
        assert p[0][3] == [0, 1, 2, 3]
        assert p == dict(nx.all_pairs_dijkstra_path(self.cycle))
        p = nx.shortest_path(self.grid, weight='weight')
        validate_grid_path(4, 4, 1, 12, p[1][12])
        p = nx.shortest_path(self.cycle, weight='weight', method='dijkstra')
        assert p[0][3] == [0, 1, 2, 3]
        assert p == dict(nx.all_pairs_dijkstra_path(self.cycle))
        p = nx.shortest_path(self.cycle, weight='weight', method='bellman-ford')
        assert p[0][3] == [0, 1, 2, 3]
        assert p == dict(nx.all_pairs_bellman_ford_path(self.cycle))

    def test_all_pairs_shortest_path_length(self):
        ans = dict(nx.shortest_path_length(self.cycle))
        assert ans[0] == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
        assert ans == dict(nx.all_pairs_shortest_path_length(self.cycle))
        ans = dict(nx.shortest_path_length(self.grid))
        assert ans[1][16] == 6
        ans = dict(nx.shortest_path_length(self.cycle, weight='weight'))
        assert ans[0] == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
        assert ans == dict(nx.all_pairs_dijkstra_path_length(self.cycle))
        ans = dict(nx.shortest_path_length(self.grid, weight='weight'))
        assert ans[1][16] == 6
        ans = dict(nx.shortest_path_length(self.cycle, weight='weight', method='dijkstra'))
        assert ans[0] == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
        assert ans == dict(nx.all_pairs_dijkstra_path_length(self.cycle))
        ans = dict(nx.shortest_path_length(self.cycle, weight='weight', method='bellman-ford'))
        assert ans[0] == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
        assert ans == dict(nx.all_pairs_bellman_ford_path_length(self.cycle))

    def test_all_pairs_all_shortest_paths(self):
        ans = dict(nx.all_pairs_all_shortest_paths(nx.cycle_graph(4)))
        assert sorted(ans[1][3]) == [[1, 0, 3], [1, 2, 3]]
        ans = dict(nx.all_pairs_all_shortest_paths(nx.cycle_graph(4)), weight='weight')
        assert sorted(ans[1][3]) == [[1, 0, 3], [1, 2, 3]]
        ans = dict(nx.all_pairs_all_shortest_paths(nx.cycle_graph(4)), method='bellman-ford', weight='weight')
        assert sorted(ans[1][3]) == [[1, 0, 3], [1, 2, 3]]
        G = nx.cycle_graph(4)
        G.add_node(4)
        ans = dict(nx.all_pairs_all_shortest_paths(G))
        assert sorted(ans[4][4]) == [[4]]

    def test_has_path(self):
        G = nx.Graph()
        nx.add_path(G, range(3))
        nx.add_path(G, range(3, 5))
        assert nx.has_path(G, 0, 2)
        assert not nx.has_path(G, 0, 4)

    def test_all_shortest_paths(self):
        G = nx.Graph()
        nx.add_path(G, [0, 1, 2, 3])
        nx.add_path(G, [0, 10, 20, 3])
        assert [[0, 1, 2, 3], [0, 10, 20, 3]] == sorted(nx.all_shortest_paths(G, 0, 3))
        G = nx.Graph()
        nx.add_path(G, [0, 1, 2, 3])
        nx.add_path(G, [0, 10, 20, 3])
        assert [[0, 1, 2, 3], [0, 10, 20, 3]] == sorted(nx.all_shortest_paths(G, 0, 3, weight='weight'))
        G = nx.Graph()
        nx.add_path(G, [0, 1, 2, 3])
        nx.add_path(G, [0, 10, 20, 3])
        assert [[0, 1, 2, 3], [0, 10, 20, 3]] == sorted(nx.all_shortest_paths(G, 0, 3, weight='weight', method='dijkstra'))
        G = nx.Graph()
        nx.add_path(G, [0, 1, 2, 3])
        nx.add_path(G, [0, 10, 20, 3])
        assert [[0, 1, 2, 3], [0, 10, 20, 3]] == sorted(nx.all_shortest_paths(G, 0, 3, weight='weight', method='bellman-ford'))

    def test_all_shortest_paths_raise(self):
        with pytest.raises(nx.NetworkXNoPath):
            G = nx.path_graph(4)
            G.add_node(4)
            list(nx.all_shortest_paths(G, 0, 4))

    def test_bad_method(self):
        with pytest.raises(ValueError):
            G = nx.path_graph(2)
            list(nx.all_shortest_paths(G, 0, 1, weight='weight', method='SPAM'))

    def test_single_source_all_shortest_paths_bad_method(self):
        with pytest.raises(ValueError):
            G = nx.path_graph(2)
            dict(nx.single_source_all_shortest_paths(G, 0, weight='weight', method='SPAM'))

    def test_all_shortest_paths_zero_weight_edge(self):
        g = nx.Graph()
        nx.add_path(g, [0, 1, 3])
        nx.add_path(g, [0, 1, 2, 3])
        g.edges[1, 2]['weight'] = 0
        paths30d = list(nx.all_shortest_paths(g, 3, 0, weight='weight', method='dijkstra'))
        paths03d = list(nx.all_shortest_paths(g, 0, 3, weight='weight', method='dijkstra'))
        paths30b = list(nx.all_shortest_paths(g, 3, 0, weight='weight', method='bellman-ford'))
        paths03b = list(nx.all_shortest_paths(g, 0, 3, weight='weight', method='bellman-ford'))
        assert sorted(paths03d) == sorted((p[::-1] for p in paths30d))
        assert sorted(paths03d) == sorted((p[::-1] for p in paths30b))
        assert sorted(paths03b) == sorted((p[::-1] for p in paths30b))