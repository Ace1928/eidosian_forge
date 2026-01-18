import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
class TestOutMultiEdgeView(TestMultiEdgeView):

    @classmethod
    def setup_class(cls):
        cls.G = nx.path_graph(9, nx.MultiDiGraph())
        cls.G.add_edge(1, 2, key=3, foo='bar')
        cls.eview = nx.reportviews.OutMultiEdgeView

    def modify_edge(self, G, e, **kwds):
        if len(e) == 2:
            e = e + (0,)
        G._adj[e[0]][e[1]][e[2]].update(kwds)

    def test_repr(self):
        ev = self.eview(self.G)
        rep = 'OutMultiEdgeView([(0, 1, 0), (1, 2, 0), (1, 2, 3), (2, 3, 0),' + ' (3, 4, 0), (4, 5, 0), (5, 6, 0), (6, 7, 0), (7, 8, 0)])'
        assert repr(ev) == rep

    def test_contains_with_nbunch(self):
        ev = self.eview(self.G)
        evn = ev(nbunch=[0, 2])
        assert (0, 1) in evn
        assert (1, 2) not in evn
        assert (2, 3) in evn
        assert (3, 4) not in evn
        assert (4, 5) not in evn
        assert (5, 6) not in evn
        assert (7, 8) not in evn
        assert (8, 9) not in evn