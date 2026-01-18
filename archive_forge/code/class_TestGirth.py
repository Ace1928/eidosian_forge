from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
class TestGirth:

    @pytest.mark.parametrize(('G', 'expected'), ((nx.chvatal_graph(), 4), (nx.tutte_graph(), 4), (nx.petersen_graph(), 5), (nx.heawood_graph(), 6), (nx.pappus_graph(), 6), (nx.random_tree(10, seed=42), inf), (nx.empty_graph(10), inf), (nx.Graph(chain(cycle_edges(range(5)), cycle_edges(range(6, 10)))), 4), (nx.Graph([(0, 6), (0, 8), (0, 9), (1, 8), (2, 8), (2, 9), (4, 9), (5, 9), (6, 8), (6, 9), (7, 8)]), 3)))
    def test_girth(self, G, expected):
        assert nx.girth(G) == expected