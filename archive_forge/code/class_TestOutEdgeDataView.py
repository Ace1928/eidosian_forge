import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
class TestOutEdgeDataView(TestEdgeDataView):

    @classmethod
    def setup_class(cls):
        cls.G = nx.path_graph(9, create_using=nx.DiGraph())
        cls.eview = nx.reportviews.OutEdgeView

    def test_repr(self):
        ev = self.eview(self.G)(data=True)
        rep = 'OutEdgeDataView([(0, 1, {}), (1, 2, {}), ' + '(2, 3, {}), (3, 4, {}), ' + '(4, 5, {}), (5, 6, {}), ' + '(6, 7, {}), (7, 8, {})])'
        assert repr(ev) == rep

    def test_len(self):
        evr = self.eview(self.G)
        ev = evr(data='foo')
        assert len(ev) == 8
        assert len(evr(1)) == 1
        assert len(evr([1, 2, 3])) == 3
        assert len(self.G.edges(1)) == 1
        assert len(self.G.edges()) == 8
        assert len(self.G.edges) == 8
        H = self.G.copy()
        H.add_edge(1, 1)
        assert len(H.edges(1)) == 2
        assert len(H.edges()) == 9
        assert len(H.edges) == 9

    def test_contains_with_nbunch(self):
        evr = self.eview(self.G)
        ev = evr(nbunch=[0, 2])
        assert (0, 1) in ev
        assert (1, 2) not in ev
        assert (2, 3) in ev
        assert (3, 4) not in ev
        assert (4, 5) not in ev
        assert (5, 6) not in ev
        assert (7, 8) not in ev
        assert (8, 9) not in ev