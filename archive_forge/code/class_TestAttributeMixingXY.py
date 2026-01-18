import networkx as nx
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
class TestAttributeMixingXY(BaseTestAttributeMixing):

    def test_node_attribute_xy_undirected(self):
        attrxy = sorted(nx.node_attribute_xy(self.G, 'fish'))
        attrxy_result = sorted([('one', 'one'), ('one', 'one'), ('two', 'two'), ('two', 'two'), ('one', 'red'), ('red', 'one'), ('blue', 'two'), ('two', 'blue')])
        assert attrxy == attrxy_result

    def test_node_attribute_xy_undirected_nodes(self):
        attrxy = sorted(nx.node_attribute_xy(self.G, 'fish', nodes=['one', 'yellow']))
        attrxy_result = sorted([])
        assert attrxy == attrxy_result

    def test_node_attribute_xy_directed(self):
        attrxy = sorted(nx.node_attribute_xy(self.D, 'fish'))
        attrxy_result = sorted([('one', 'one'), ('two', 'two'), ('one', 'red'), ('two', 'blue')])
        assert attrxy == attrxy_result

    def test_node_attribute_xy_multigraph(self):
        attrxy = sorted(nx.node_attribute_xy(self.M, 'fish'))
        attrxy_result = [('one', 'one'), ('one', 'one'), ('one', 'one'), ('one', 'one'), ('two', 'two'), ('two', 'two')]
        assert attrxy == attrxy_result

    def test_node_attribute_xy_selfloop(self):
        attrxy = sorted(nx.node_attribute_xy(self.S, 'fish'))
        attrxy_result = [('one', 'one'), ('two', 'two')]
        assert attrxy == attrxy_result