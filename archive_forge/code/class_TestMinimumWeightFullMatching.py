import itertools
import pytest
import networkx as nx
from networkx.algorithms.bipartite.matching import (
class TestMinimumWeightFullMatching:

    @classmethod
    def setup_class(cls):
        pytest.importorskip('scipy')

    def test_minimum_weight_full_matching_incomplete_graph(self):
        B = nx.Graph()
        B.add_nodes_from([1, 2], bipartite=0)
        B.add_nodes_from([3, 4], bipartite=1)
        B.add_edge(1, 4, weight=100)
        B.add_edge(2, 3, weight=100)
        B.add_edge(2, 4, weight=50)
        matching = minimum_weight_full_matching(B)
        assert matching == {1: 4, 2: 3, 4: 1, 3: 2}

    def test_minimum_weight_full_matching_with_no_full_matching(self):
        B = nx.Graph()
        B.add_nodes_from([1, 2, 3], bipartite=0)
        B.add_nodes_from([4, 5, 6], bipartite=1)
        B.add_edge(1, 4, weight=100)
        B.add_edge(2, 4, weight=100)
        B.add_edge(3, 4, weight=50)
        B.add_edge(3, 5, weight=50)
        B.add_edge(3, 6, weight=50)
        with pytest.raises(ValueError):
            minimum_weight_full_matching(B)

    def test_minimum_weight_full_matching_square(self):
        G = nx.complete_bipartite_graph(3, 3)
        G.add_edge(0, 3, weight=400)
        G.add_edge(0, 4, weight=150)
        G.add_edge(0, 5, weight=400)
        G.add_edge(1, 3, weight=400)
        G.add_edge(1, 4, weight=450)
        G.add_edge(1, 5, weight=600)
        G.add_edge(2, 3, weight=300)
        G.add_edge(2, 4, weight=225)
        G.add_edge(2, 5, weight=300)
        matching = minimum_weight_full_matching(G)
        assert matching == {0: 4, 1: 3, 2: 5, 4: 0, 3: 1, 5: 2}

    def test_minimum_weight_full_matching_smaller_left(self):
        G = nx.complete_bipartite_graph(3, 4)
        G.add_edge(0, 3, weight=400)
        G.add_edge(0, 4, weight=150)
        G.add_edge(0, 5, weight=400)
        G.add_edge(0, 6, weight=1)
        G.add_edge(1, 3, weight=400)
        G.add_edge(1, 4, weight=450)
        G.add_edge(1, 5, weight=600)
        G.add_edge(1, 6, weight=2)
        G.add_edge(2, 3, weight=300)
        G.add_edge(2, 4, weight=225)
        G.add_edge(2, 5, weight=290)
        G.add_edge(2, 6, weight=3)
        matching = minimum_weight_full_matching(G)
        assert matching == {0: 4, 1: 6, 2: 5, 4: 0, 5: 2, 6: 1}

    def test_minimum_weight_full_matching_smaller_top_nodes_right(self):
        G = nx.complete_bipartite_graph(3, 4)
        G.add_edge(0, 3, weight=400)
        G.add_edge(0, 4, weight=150)
        G.add_edge(0, 5, weight=400)
        G.add_edge(0, 6, weight=1)
        G.add_edge(1, 3, weight=400)
        G.add_edge(1, 4, weight=450)
        G.add_edge(1, 5, weight=600)
        G.add_edge(1, 6, weight=2)
        G.add_edge(2, 3, weight=300)
        G.add_edge(2, 4, weight=225)
        G.add_edge(2, 5, weight=290)
        G.add_edge(2, 6, weight=3)
        matching = minimum_weight_full_matching(G, top_nodes=[3, 4, 5, 6])
        assert matching == {0: 4, 1: 6, 2: 5, 4: 0, 5: 2, 6: 1}

    def test_minimum_weight_full_matching_smaller_right(self):
        G = nx.complete_bipartite_graph(4, 3)
        G.add_edge(0, 4, weight=400)
        G.add_edge(0, 5, weight=400)
        G.add_edge(0, 6, weight=300)
        G.add_edge(1, 4, weight=150)
        G.add_edge(1, 5, weight=450)
        G.add_edge(1, 6, weight=225)
        G.add_edge(2, 4, weight=400)
        G.add_edge(2, 5, weight=600)
        G.add_edge(2, 6, weight=290)
        G.add_edge(3, 4, weight=1)
        G.add_edge(3, 5, weight=2)
        G.add_edge(3, 6, weight=3)
        matching = minimum_weight_full_matching(G)
        assert matching == {1: 4, 2: 6, 3: 5, 4: 1, 5: 3, 6: 2}

    def test_minimum_weight_full_matching_negative_weights(self):
        G = nx.complete_bipartite_graph(2, 2)
        G.add_edge(0, 2, weight=-2)
        G.add_edge(0, 3, weight=0.2)
        G.add_edge(1, 2, weight=-2)
        G.add_edge(1, 3, weight=0.3)
        matching = minimum_weight_full_matching(G)
        assert matching == {0: 3, 1: 2, 2: 1, 3: 0}

    def test_minimum_weight_full_matching_different_weight_key(self):
        G = nx.complete_bipartite_graph(2, 2)
        G.add_edge(0, 2, mass=2)
        G.add_edge(0, 3, mass=0.2)
        G.add_edge(1, 2, mass=1)
        G.add_edge(1, 3, mass=2)
        matching = minimum_weight_full_matching(G, weight='mass')
        assert matching == {0: 3, 1: 2, 2: 1, 3: 0}