import pytest
import networkx as nx
class TestUnweightedPath:

    @classmethod
    def setup_class(cls):
        from networkx import convert_node_labels_to_integers as cnlti
        cls.grid = cnlti(nx.grid_2d_graph(4, 4), first_label=1, ordering='sorted')
        cls.cycle = nx.cycle_graph(7)
        cls.directed_cycle = nx.cycle_graph(7, create_using=nx.DiGraph())

    def test_bidirectional_shortest_path(self):
        assert nx.bidirectional_shortest_path(self.cycle, 0, 3) == [0, 1, 2, 3]
        assert nx.bidirectional_shortest_path(self.cycle, 0, 4) == [0, 6, 5, 4]
        validate_grid_path(4, 4, 1, 12, nx.bidirectional_shortest_path(self.grid, 1, 12))
        assert nx.bidirectional_shortest_path(self.directed_cycle, 0, 3) == [0, 1, 2, 3]
        assert nx.bidirectional_shortest_path(self.cycle, 3, 3) == [3]

    @pytest.mark.parametrize(('src', 'tgt'), ((8, 3), (3, 8), (8, 10), (8, 8)))
    def test_bidirectional_shortest_path_src_tgt_not_in_graph(self, src, tgt):
        with pytest.raises(nx.NodeNotFound, match=f'Either source {src} or target {tgt} is not in G'):
            nx.bidirectional_shortest_path(self.cycle, src, tgt)

    def test_shortest_path_length(self):
        assert nx.shortest_path_length(self.cycle, 0, 3) == 3
        assert nx.shortest_path_length(self.grid, 1, 12) == 5
        assert nx.shortest_path_length(self.directed_cycle, 0, 4) == 4
        assert nx.shortest_path_length(self.cycle, 0, 3, weight=True) == 3
        assert nx.shortest_path_length(self.grid, 1, 12, weight=True) == 5
        assert nx.shortest_path_length(self.directed_cycle, 0, 4, weight=True) == 4

    def test_single_source_shortest_path(self):
        p = nx.single_source_shortest_path(self.directed_cycle, 3)
        assert p[0] == [3, 4, 5, 6, 0]
        p = nx.single_source_shortest_path(self.cycle, 0)
        assert p[3] == [0, 1, 2, 3]
        p = nx.single_source_shortest_path(self.cycle, 0, cutoff=0)
        assert p == {0: [0]}

    def test_single_source_shortest_path_length(self):
        pl = nx.single_source_shortest_path_length
        lengths = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
        assert dict(pl(self.cycle, 0)) == lengths
        lengths = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
        assert dict(pl(self.directed_cycle, 0)) == lengths

    def test_single_target_shortest_path(self):
        p = nx.single_target_shortest_path(self.directed_cycle, 0)
        assert p[3] == [3, 4, 5, 6, 0]
        p = nx.single_target_shortest_path(self.cycle, 0)
        assert p[3] == [3, 2, 1, 0]
        p = nx.single_target_shortest_path(self.cycle, 0, cutoff=0)
        assert p == {0: [0]}
        target = 8
        with pytest.raises(nx.NodeNotFound, match=f'Target {target} not in G'):
            nx.single_target_shortest_path(self.cycle, target)

    def test_single_target_shortest_path_length(self):
        pl = nx.single_target_shortest_path_length
        lengths = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
        assert dict(pl(self.cycle, 0)) == lengths
        lengths = {0: 0, 1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
        assert dict(pl(self.directed_cycle, 0)) == lengths
        target = 8
        with pytest.raises(nx.NodeNotFound, match=f'Target {target} is not in G'):
            nx.single_target_shortest_path_length(self.cycle, target)

    def test_all_pairs_shortest_path(self):
        p = dict(nx.all_pairs_shortest_path(self.cycle))
        assert p[0][3] == [0, 1, 2, 3]
        p = dict(nx.all_pairs_shortest_path(self.grid))
        validate_grid_path(4, 4, 1, 12, p[1][12])

    def test_all_pairs_shortest_path_length(self):
        l = dict(nx.all_pairs_shortest_path_length(self.cycle))
        assert l[0] == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
        l = dict(nx.all_pairs_shortest_path_length(self.grid))
        assert l[1][16] == 6

    def test_predecessor_path(self):
        G = nx.path_graph(4)
        assert nx.predecessor(G, 0) == {0: [], 1: [0], 2: [1], 3: [2]}
        assert nx.predecessor(G, 0, 3) == [2]

    def test_predecessor_cycle(self):
        G = nx.cycle_graph(4)
        pred = nx.predecessor(G, 0)
        assert pred[0] == []
        assert pred[1] == [0]
        assert pred[2] in [[1, 3], [3, 1]]
        assert pred[3] == [0]

    def test_predecessor_cutoff(self):
        G = nx.path_graph(4)
        p = nx.predecessor(G, 0, 3)
        assert 4 not in p

    def test_predecessor_target(self):
        G = nx.path_graph(4)
        p = nx.predecessor(G, 0, 3)
        assert p == [2]
        p = nx.predecessor(G, 0, 3, cutoff=2)
        assert p == []
        p, s = nx.predecessor(G, 0, 3, return_seen=True)
        assert p == [2]
        assert s == 3
        p, s = nx.predecessor(G, 0, 3, cutoff=2, return_seen=True)
        assert p == []
        assert s == -1

    def test_predecessor_missing_source(self):
        source = 8
        with pytest.raises(nx.NodeNotFound, match=f'Source {source} not in G'):
            nx.predecessor(self.cycle, source)