import pickle
import pytest
import networkx as nx
class TestUnionAdjacency:

    def setup_method(self):
        dd = {'color': 'blue', 'weight': 1.2}
        self.nd = {0: dd, 1: {}, 2: {'color': 1}}
        self.s = {3: self.nd, 0: {}, 1: {}, 2: {3: {'color': 1}}}
        self.p = {3: {}, 0: {3: dd}, 1: {0: {}}, 2: {1: {'color': 1}}}
        self.adjview = nx.classes.coreviews.UnionAdjacency(self.s, self.p)

    def test_pickle(self):
        view = self.adjview
        pview = pickle.loads(pickle.dumps(view, -1))
        assert view == pview
        assert view.__slots__ == pview.__slots__

    def test_len(self):
        assert len(self.adjview) == len(self.s)

    def test_iter(self):
        assert sorted(self.adjview) == sorted(self.s)

    def test_getitem(self):
        assert self.adjview[1] is not self.s[1]
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

    def test_str(self):
        out = str(dict(self.adjview))
        assert str(self.adjview) == out

    def test_repr(self):
        clsname = self.adjview.__class__.__name__
        out = f'{clsname}({self.s}, {self.p})'
        assert repr(self.adjview) == out