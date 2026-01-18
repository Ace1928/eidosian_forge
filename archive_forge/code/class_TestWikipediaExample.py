import pytest
import networkx as nx
from networkx.algorithms import isomorphism as iso
class TestWikipediaExample:
    g1edges = [['a', 'g'], ['a', 'h'], ['a', 'i'], ['b', 'g'], ['b', 'h'], ['b', 'j'], ['c', 'g'], ['c', 'i'], ['c', 'j'], ['d', 'h'], ['d', 'i'], ['d', 'j']]
    g2edges = [[1, 2], [2, 3], [3, 4], [4, 1], [5, 6], [6, 7], [7, 8], [8, 5], [1, 5], [2, 6], [3, 7], [4, 8]]

    def test_graph(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        g1.add_edges_from(self.g1edges)
        g2.add_edges_from(self.g2edges)
        gm = iso.ISMAGS(g1, g2)
        assert gm.is_isomorphic()