import pickle
import pytest
import networkx as nx
class TestAtlasView:

    def setup_method(self):
        self.d = {0: {'color': 'blue', 'weight': 1.2}, 1: {}, 2: {'color': 1}}
        self.av = nx.classes.coreviews.AtlasView(self.d)

    def test_pickle(self):
        view = self.av
        pview = pickle.loads(pickle.dumps(view, -1))
        assert view == pview
        assert view.__slots__ == pview.__slots__
        pview = pickle.loads(pickle.dumps(view))
        assert view == pview
        assert view.__slots__ == pview.__slots__

    def test_len(self):
        assert len(self.av) == len(self.d)

    def test_iter(self):
        assert list(self.av) == list(self.d)

    def test_getitem(self):
        assert self.av[1] is self.d[1]
        assert self.av[2]['color'] == 1
        pytest.raises(KeyError, self.av.__getitem__, 3)

    def test_copy(self):
        avcopy = self.av.copy()
        assert avcopy[0] == self.av[0]
        assert avcopy == self.av
        assert avcopy[0] is not self.av[0]
        assert avcopy is not self.av
        avcopy[5] = {}
        assert avcopy != self.av
        avcopy[0]['ht'] = 4
        assert avcopy[0] != self.av[0]
        self.av[0]['ht'] = 4
        assert avcopy[0] == self.av[0]
        del self.av[0]['ht']
        assert not hasattr(self.av, '__setitem__')

    def test_items(self):
        assert sorted(self.av.items()) == sorted(self.d.items())

    def test_str(self):
        out = str(self.d)
        assert str(self.av) == out

    def test_repr(self):
        out = 'AtlasView(' + str(self.d) + ')'
        assert repr(self.av) == out