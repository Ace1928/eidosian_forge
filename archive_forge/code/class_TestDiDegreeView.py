import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
class TestDiDegreeView(TestDegreeView):
    GRAPH = nx.DiGraph
    dview = nx.reportviews.DiDegreeView

    def test_repr(self):
        dv = self.G.degree()
        rep = 'DiDegreeView({0: 1, 1: 3, 2: 2, 3: 3, 4: 2, 5: 1})'
        assert repr(dv) == rep