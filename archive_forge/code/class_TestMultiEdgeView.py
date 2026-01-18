import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
class TestMultiEdgeView(TestEdgeView):

    @classmethod
    def setup_class(cls):
        cls.G = nx.path_graph(9, nx.MultiGraph())
        cls.G.add_edge(1, 2, key=3, foo='bar')
        cls.eview = nx.reportviews.MultiEdgeView

    def modify_edge(self, G, e, **kwds):
        if len(e) == 2:
            e = e + (0,)
        G._adj[e[0]][e[1]][e[2]].update(kwds)

    def test_str(self):
        ev = self.eview(self.G)
        replist = [(n, n + 1, 0) for n in range(8)]
        replist.insert(2, (1, 2, 3))
        rep = str(replist)
        assert str(ev) == rep

    def test_getitem(self):
        G = self.G.copy()
        ev = G.edges
        G.edges[0, 1, 0]['foo'] = 'bar'
        assert ev[0, 1, 0] == {'foo': 'bar'}
        with pytest.raises(nx.NetworkXError):
            G.edges[0:5]

    def test_repr(self):
        ev = self.eview(self.G)
        rep = 'MultiEdgeView([(0, 1, 0), (1, 2, 0), (1, 2, 3), (2, 3, 0), ' + '(3, 4, 0), (4, 5, 0), (5, 6, 0), (6, 7, 0), (7, 8, 0)])'
        assert repr(ev) == rep

    def test_call(self):
        ev = self.eview(self.G)
        assert id(ev) == id(ev(keys=True))
        assert id(ev) == id(ev(data=False, keys=True))
        assert id(ev) != id(ev(keys=False))
        assert id(ev) != id(ev(data=True))
        assert id(ev) != id(ev(nbunch=1))

    def test_data(self):
        ev = self.eview(self.G)
        assert id(ev) != id(ev.data())
        assert id(ev) == id(ev.data(data=False, keys=True))
        assert id(ev) != id(ev.data(keys=False))
        assert id(ev) != id(ev.data(data=True))
        assert id(ev) != id(ev.data(nbunch=1))

    def test_iter(self):
        ev = self.eview(self.G)
        for u, v, k in ev:
            pass
        iev = iter(ev)
        assert next(iev) == (0, 1, 0)
        assert iter(ev) != ev
        assert iter(iev) == iev

    def test_iterkeys(self):
        G = self.G
        evr = self.eview(G)
        ev = evr(keys=True)
        for u, v, k in ev:
            pass
        assert k == 0
        ev = evr(keys=True, data='foo', default=1)
        for u, v, k, wt in ev:
            pass
        assert wt == 1
        self.modify_edge(G, (2, 3, 0), foo='bar')
        ev = evr(keys=True, data=True)
        for e in ev:
            assert len(e) == 4
            print('edge:', e)
            if set(e[:2]) == {2, 3}:
                print(self.G._adj[2][3])
                assert e[2] == 0
                assert e[3] == {'foo': 'bar'}
                checked = True
            elif set(e[:3]) == {1, 2, 3}:
                assert e[2] == 3
                assert e[3] == {'foo': 'bar'}
                checked_multi = True
            else:
                assert e[2] == 0
                assert e[3] == {}
        assert checked
        assert checked_multi
        ev = evr(keys=True, data='foo', default=1)
        for e in ev:
            if set(e[:2]) == {1, 2} and e[2] == 3:
                assert e[3] == 'bar'
            if set(e[:2]) == {1, 2} and e[2] == 0:
                assert e[3] == 1
            if set(e[:2]) == {2, 3}:
                assert e[2] == 0
                assert e[3] == 'bar'
                assert len(e) == 4
                checked_wt = True
        assert checked_wt
        ev = evr(keys=True)
        for e in ev:
            assert len(e) == 3
        elist = sorted([(i, i + 1, 0) for i in range(8)] + [(1, 2, 3)])
        assert sorted(ev) == elist
        ev = evr((1, 2), 'foo', keys=True, default=1)
        with pytest.raises(TypeError):
            evr((1, 2), 'foo', True, 1)
        with pytest.raises(TypeError):
            evr((1, 2), 'foo', True, default=1)
        for e in ev:
            if set(e[:2]) == {1, 2}:
                assert e[2] in {0, 3}
                if e[2] == 3:
                    assert e[3] == 'bar'
                else:
                    assert e[3] == 1
        if G.is_directed():
            assert len(list(ev)) == 3
        else:
            assert len(list(ev)) == 4

    def test_or(self):
        ev = self.eview(self.G)
        some_edges = {(0, 1, 0), (1, 0, 0), (0, 2, 0)}
        result = {(n, n + 1, 0) for n in range(8)}
        result.update(some_edges)
        result.update({(1, 2, 3)})
        assert ev | some_edges == result
        assert some_edges | ev == result

    def test_sub(self):
        ev = self.eview(self.G)
        some_edges = {(0, 1, 0), (1, 0, 0), (0, 2, 0)}
        result = {(n, n + 1, 0) for n in range(8)}
        result.remove((0, 1, 0))
        result.update({(1, 2, 3)})
        assert ev - some_edges, result
        assert some_edges - ev, result

    def test_xor(self):
        ev = self.eview(self.G)
        some_edges = {(0, 1, 0), (1, 0, 0), (0, 2, 0)}
        if self.G.is_directed():
            result = {(n, n + 1, 0) for n in range(1, 8)}
            result.update({(1, 0, 0), (0, 2, 0), (1, 2, 3)})
            assert ev ^ some_edges == result
            assert some_edges ^ ev == result
        else:
            result = {(n, n + 1, 0) for n in range(1, 8)}
            result.update({(0, 2, 0), (1, 2, 3)})
            assert ev ^ some_edges == result
            assert some_edges ^ ev == result

    def test_and(self):
        ev = self.eview(self.G)
        some_edges = {(0, 1, 0), (1, 0, 0), (0, 2, 0)}
        if self.G.is_directed():
            assert ev & some_edges == {(0, 1, 0)}
            assert some_edges & ev == {(0, 1, 0)}
        else:
            assert ev & some_edges == {(0, 1, 0), (1, 0, 0)}
            assert some_edges & ev == {(0, 1, 0), (1, 0, 0)}

    def test_contains_with_nbunch(self):
        ev = self.eview(self.G)
        evn = ev(nbunch=[0, 2])
        assert (0, 1) in evn
        assert (1, 2) in evn
        assert (2, 3) in evn
        assert (3, 4) not in evn
        assert (4, 5) not in evn
        assert (5, 6) not in evn
        assert (7, 8) not in evn
        assert (8, 9) not in evn