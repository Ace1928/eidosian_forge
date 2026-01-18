import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
class TestInEdgeView(TestEdgeView):

    @classmethod
    def setup_class(cls):
        cls.G = nx.path_graph(9, nx.DiGraph())
        cls.eview = nx.reportviews.InEdgeView

    def test_repr(self):
        ev = self.eview(self.G)
        rep = 'InEdgeView([(0, 1), (1, 2), (2, 3), (3, 4), ' + '(4, 5), (5, 6), (6, 7), (7, 8)])'
        assert repr(ev) == rep

    def test_contains_with_nbunch(self):
        ev = self.eview(self.G)
        evn = ev(nbunch=[0, 2])
        assert (0, 1) not in evn
        assert (1, 2) in evn
        assert (2, 3) not in evn
        assert (3, 4) not in evn
        assert (4, 5) not in evn
        assert (5, 6) not in evn
        assert (7, 8) not in evn
        assert (8, 9) not in evn