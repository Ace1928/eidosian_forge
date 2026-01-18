import math
from operator import eq
import networkx as nx
import networkx.algorithms.isomorphism as iso
class TestNodeMatch_Graph:

    def setup_method(self):
        self.g1 = nx.Graph()
        self.g2 = nx.Graph()
        self.build()

    def build(self):
        self.nm = iso.categorical_node_match('color', '')
        self.em = iso.numerical_edge_match('weight', 1)
        self.g1.add_node('A', color='red')
        self.g2.add_node('C', color='blue')
        self.g1.add_edge('A', 'B', weight=1)
        self.g2.add_edge('C', 'D', weight=1)

    def test_noweight_nocolor(self):
        assert nx.is_isomorphic(self.g1, self.g2)

    def test_color1(self):
        assert not nx.is_isomorphic(self.g1, self.g2, node_match=self.nm)

    def test_color2(self):
        self.g1.nodes['A']['color'] = 'blue'
        assert nx.is_isomorphic(self.g1, self.g2, node_match=self.nm)

    def test_weight1(self):
        assert nx.is_isomorphic(self.g1, self.g2, edge_match=self.em)

    def test_weight2(self):
        self.g1.add_edge('A', 'B', weight=2)
        assert not nx.is_isomorphic(self.g1, self.g2, edge_match=self.em)

    def test_colorsandweights1(self):
        iso = nx.is_isomorphic(self.g1, self.g2, node_match=self.nm, edge_match=self.em)
        assert not iso

    def test_colorsandweights2(self):
        self.g1.nodes['A']['color'] = 'blue'
        iso = nx.is_isomorphic(self.g1, self.g2, node_match=self.nm, edge_match=self.em)
        assert iso

    def test_colorsandweights3(self):
        self.g1.add_edge('A', 'B', weight=2)
        assert not nx.is_isomorphic(self.g1, self.g2, node_match=self.nm, edge_match=self.em)