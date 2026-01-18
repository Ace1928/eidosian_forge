import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
class TestNodeDataView:

    @classmethod
    def setup_class(cls):
        cls.G = nx.path_graph(9)
        cls.nv = NodeDataView(cls.G)
        cls.ndv = cls.G.nodes.data(True)
        cls.nwv = cls.G.nodes.data('foo')

    def test_viewtype(self):
        nv = self.G.nodes
        ndvfalse = nv.data(False)
        assert nv is ndvfalse
        assert nv is not self.ndv

    def test_pickle(self):
        import pickle
        nv = self.nv
        pnv = pickle.loads(pickle.dumps(nv, -1))
        assert nv == pnv
        assert nv.__slots__ == pnv.__slots__

    def test_str(self):
        msg = str([(n, {}) for n in range(9)])
        assert str(self.ndv) == msg

    def test_repr(self):
        expected = 'NodeDataView((0, 1, 2, 3, 4, 5, 6, 7, 8))'
        assert repr(self.nv) == expected
        expected = 'NodeDataView({0: {}, 1: {}, 2: {}, 3: {}, ' + '4: {}, 5: {}, 6: {}, 7: {}, 8: {}})'
        assert repr(self.ndv) == expected
        expected = 'NodeDataView({0: None, 1: None, 2: None, 3: None, 4: None, ' + "5: None, 6: None, 7: None, 8: None}, data='foo')"
        assert repr(self.nwv) == expected

    def test_contains(self):
        G = self.G.copy()
        nv = G.nodes.data()
        nwv = G.nodes.data('foo')
        G.nodes[3]['foo'] = 'bar'
        assert (7, {}) in nv
        assert (3, {'foo': 'bar'}) in nv
        assert (3, 'bar') in nwv
        assert (7, None) in nwv
        nwv_def = G.nodes(data='foo', default='biz')
        assert (7, 'biz') in nwv_def
        assert (3, 'bar') in nwv_def

    def test_getitem(self):
        G = self.G.copy()
        nv = G.nodes
        G.nodes[3]['foo'] = 'bar'
        assert nv[3] == {'foo': 'bar'}
        nwv_def = G.nodes(data='foo', default='biz')
        assert nwv_def[7], 'biz'
        assert nwv_def[3] == 'bar'
        with pytest.raises(nx.NetworkXError):
            G.nodes.data()[0:5]

    def test_iter(self):
        G = self.G.copy()
        nv = G.nodes.data()
        ndv = G.nodes.data(True)
        nwv = G.nodes.data('foo')
        for i, (n, d) in enumerate(nv):
            assert i == n
            assert d == {}
        inv = iter(nv)
        assert next(inv) == (0, {})
        G.nodes[3]['foo'] = 'bar'
        for n, d in nv:
            if n == 3:
                assert d == {'foo': 'bar'}
            else:
                assert d == {}
        for n, d in ndv:
            if n == 3:
                assert d == {'foo': 'bar'}
            else:
                assert d == {}
        for n, d in nwv:
            if n == 3:
                assert d == 'bar'
            else:
                assert d is None
        for n, d in G.nodes.data('foo', default=1):
            if n == 3:
                assert d == 'bar'
            else:
                assert d == 1