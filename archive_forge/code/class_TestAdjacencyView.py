import pickle
import pytest
import networkx as nx
class TestAdjacencyView:

    def setup_method(self):
        dd = {'color': 'blue', 'weight': 1.2}
        self.nd = {0: dd, 1: {}, 2: {'color': 1}}
        self.adj = {3: self.nd, 0: {3: dd}, 1: {}, 2: {3: {'color': 1}}}
        self.adjview = nx.classes.coreviews.AdjacencyView(self.adj)

    def test_pickle(self):
        view = self.adjview
        pview = pickle.loads(pickle.dumps(view, -1))
        assert view == pview
        assert view.__slots__ == pview.__slots__

    def test_len(self):
        assert len(self.adjview) == len(self.adj)

    def test_iter(self):
        assert list(self.adjview) == list(self.adj)

    def test_getitem(self):
        assert self.adjview[1] is not self.adj[1]
        assert self.adjview[3][0] is self.adjview[0][3]
        assert self.adjview[2][3]['color'] == 1
        pytest.raises(KeyError, self.adjview.__getitem__, 4)

    def test_copy(self):
        avcopy = self.adjview.copy()
        assert avcopy[0] == self.adjview[0]
        assert avcopy[0] is not self.adjview[0]
        avcopy[2][3]['ht'] = 4
        assert avcopy[2] != self.adjview[2]
        self.adjview[2][3]['ht'] = 4
        assert avcopy[2] == self.adjview[2]
        del self.adjview[2][3]['ht']
        assert not hasattr(self.adjview, '__setitem__')

    def test_items(self):
        view_items = sorted(((n, dict(d)) for n, d in self.adjview.items()))
        assert view_items == sorted(self.adj.items())

    def test_str(self):
        out = str(dict(self.adj))
        assert str(self.adjview) == out

    def test_repr(self):
        out = self.adjview.__class__.__name__ + '(' + str(self.adj) + ')'
        assert repr(self.adjview) == out