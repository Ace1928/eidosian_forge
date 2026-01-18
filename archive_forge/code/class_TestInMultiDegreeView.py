import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
class TestInMultiDegreeView(TestDegreeView):
    GRAPH = nx.MultiDiGraph
    dview = nx.reportviews.InMultiDegreeView

    def test_str(self):
        dv = self.dview(self.G)
        rep = str([(0, 0), (1, 1), (2, 1), (3, 3), (4, 1), (5, 1)])
        assert str(dv) == rep
        dv = self.G.in_degree()
        assert str(dv) == rep

    def test_repr(self):
        dv = self.G.in_degree()
        rep = 'InMultiDegreeView({0: 0, 1: 1, 2: 1, 3: 3, 4: 1, 5: 1})'
        assert repr(dv) == rep

    def test_nbunch(self):
        dv = self.dview(self.G)
        dvn = dv(0)
        assert dvn == 0
        dvn = dv([2, 3])
        assert sorted(dvn) == [(2, 1), (3, 3)]

    def test_getitem(self):
        dv = self.dview(self.G)
        assert dv[0] == 0
        assert dv[1] == 1
        assert dv[2] == 1
        assert dv[3] == 3
        dv = self.dview(self.G, weight='foo')
        assert dv[0] == 0
        assert dv[1] == 1
        assert dv[2] == 1
        assert dv[3] == 6

    def test_weight(self):
        dv = self.dview(self.G)
        dvw = dv(0, weight='foo')
        assert dvw == 0
        dvw = dv(1, weight='foo')
        assert dvw == 1
        dvw = dv([2, 3], weight='foo')
        assert sorted(dvw) == [(2, 1), (3, 6)]
        dvd = dict(dv(weight='foo'))
        assert dvd[0] == 0
        assert dvd[1] == 1
        assert dvd[2] == 1
        assert dvd[3] == 6