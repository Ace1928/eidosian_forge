from collections import UserDict
import pytest
import networkx as nx
from networkx.utils import edges_equal
from .test_graph import BaseAttrGraphTester
from .test_graph import TestGraph as _TestGraph
class TestMultiGraph(BaseMultiGraphTester, _TestGraph):

    def setup_method(self):
        self.Graph = nx.MultiGraph
        ed1, ed2, ed3 = ({0: {}}, {0: {}}, {0: {}})
        self.k3adj = {0: {1: ed1, 2: ed2}, 1: {0: ed1, 2: ed3}, 2: {0: ed2, 1: ed3}}
        self.k3edges = [(0, 1), (0, 2), (1, 2)]
        self.k3nodes = [0, 1, 2]
        self.K3 = self.Graph()
        self.K3._adj = self.k3adj
        self.K3._node = {}
        self.K3._node[0] = {}
        self.K3._node[1] = {}
        self.K3._node[2] = {}

    def test_data_input(self):
        G = self.Graph({1: [2], 2: [1]}, name='test')
        assert G.name == 'test'
        expected = [(1, {2: {0: {}}}), (2, {1: {0: {}}})]
        assert sorted(G.adj.items()) == expected

    def test_data_multigraph_input(self):
        edata0 = {'w': 200, 's': 'foo'}
        edata1 = {'w': 201, 's': 'bar'}
        keydict = {0: edata0, 1: edata1}
        dododod = {'a': {'b': keydict}}
        multiple_edge = [('a', 'b', 0, edata0), ('a', 'b', 1, edata1)]
        single_edge = [('a', 'b', 0, keydict)]
        G = self.Graph(dododod, multigraph_input=True)
        assert list(G.edges(keys=True, data=True)) == multiple_edge
        G = self.Graph(dododod, multigraph_input=None)
        assert list(G.edges(keys=True, data=True)) == multiple_edge
        G = self.Graph(dododod, multigraph_input=False)
        assert list(G.edges(keys=True, data=True)) == single_edge
        G = self.Graph(dododod, multigraph_input=True)
        H = self.Graph(nx.to_dict_of_dicts(G))
        assert nx.is_isomorphic(G, H) is True
        for mgi in [True, False]:
            H = self.Graph(nx.to_dict_of_dicts(G), multigraph_input=mgi)
            assert nx.is_isomorphic(G, H) == mgi
    etraits = {'w': 200, 's': 'foo'}
    egraphics = {'color': 'blue', 'shape': 'box'}
    edata = {'traits': etraits, 'graphics': egraphics}
    dodod1 = {'a': {'b': edata}}
    dodod2 = {'a': {'b': etraits}}
    dodod3 = {'a': {'b': {'traits': etraits, 's': 'foo'}}}
    dol = {'a': ['b']}
    multiple_edge = [('a', 'b', 'traits', etraits), ('a', 'b', 'graphics', egraphics)]
    single_edge = [('a', 'b', 0, {})]
    single_edge1 = [('a', 'b', 0, edata)]
    single_edge2 = [('a', 'b', 0, etraits)]
    single_edge3 = [('a', 'b', 0, {'traits': etraits, 's': 'foo'})]
    cases = [(dodod1, True, multiple_edge), (dodod1, False, single_edge1), (dodod2, False, single_edge2), (dodod3, False, single_edge3), (dol, False, single_edge)]

    @pytest.mark.parametrize('dod, mgi, edges', cases)
    def test_non_multigraph_input(self, dod, mgi, edges):
        G = self.Graph(dod, multigraph_input=mgi)
        assert list(G.edges(keys=True, data=True)) == edges
        G = nx.to_networkx_graph(dod, create_using=self.Graph, multigraph_input=mgi)
        assert list(G.edges(keys=True, data=True)) == edges
    mgi_none_cases = [(dodod1, multiple_edge), (dodod2, single_edge2), (dodod3, single_edge3)]

    @pytest.mark.parametrize('dod, edges', mgi_none_cases)
    def test_non_multigraph_input_mgi_none(self, dod, edges):
        G = self.Graph(dod)
        assert list(G.edges(keys=True, data=True)) == edges
    raise_cases = [dodod2, dodod3, dol]

    @pytest.mark.parametrize('dod', raise_cases)
    def test_non_multigraph_input_raise(self, dod):
        pytest.raises(nx.NetworkXError, self.Graph, dod, multigraph_input=True)
        pytest.raises(nx.NetworkXError, nx.to_networkx_graph, dod, create_using=self.Graph, multigraph_input=True)

    def test_getitem(self):
        G = self.K3
        assert G[0] == {1: {0: {}}, 2: {0: {}}}
        with pytest.raises(KeyError):
            G.__getitem__('j')
        with pytest.raises(TypeError):
            G.__getitem__(['A'])

    def test_remove_node(self):
        G = self.K3
        G.remove_node(0)
        assert G.adj == {1: {2: {0: {}}}, 2: {1: {0: {}}}}
        with pytest.raises(nx.NetworkXError):
            G.remove_node(-1)

    def test_add_edge(self):
        G = self.Graph()
        G.add_edge(0, 1)
        assert G.adj == {0: {1: {0: {}}}, 1: {0: {0: {}}}}
        G = self.Graph()
        G.add_edge(*(0, 1))
        assert G.adj == {0: {1: {0: {}}}, 1: {0: {0: {}}}}
        G = self.Graph()
        with pytest.raises(ValueError):
            G.add_edge(None, 'anything')

    def test_add_edge_conflicting_key(self):
        G = self.Graph()
        G.add_edge(0, 1, key=1)
        G.add_edge(0, 1)
        assert G.number_of_edges() == 2
        G = self.Graph()
        G.add_edges_from([(0, 1, 1, {})])
        G.add_edges_from([(0, 1)])
        assert G.number_of_edges() == 2

    def test_add_edges_from(self):
        G = self.Graph()
        G.add_edges_from([(0, 1), (0, 1, {'weight': 3})])
        assert G.adj == {0: {1: {0: {}, 1: {'weight': 3}}}, 1: {0: {0: {}, 1: {'weight': 3}}}}
        G.add_edges_from([(0, 1), (0, 1, {'weight': 3})], weight=2)
        assert G.adj == {0: {1: {0: {}, 1: {'weight': 3}, 2: {'weight': 2}, 3: {'weight': 3}}}, 1: {0: {0: {}, 1: {'weight': 3}, 2: {'weight': 2}, 3: {'weight': 3}}}}
        G = self.Graph()
        edges = [(0, 1, {'weight': 3}), (0, 1, (('weight', 2),)), (0, 1, 5), (0, 1, 's')]
        G.add_edges_from(edges)
        keydict = {0: {'weight': 3}, 1: {'weight': 2}, 5: {}, 's': {}}
        assert G._adj == {0: {1: keydict}, 1: {0: keydict}}
        with pytest.raises(nx.NetworkXError):
            G.add_edges_from([(0,)])
        with pytest.raises(nx.NetworkXError):
            G.add_edges_from([(0, 1, 2, 3, 4)])
        with pytest.raises(TypeError):
            G.add_edges_from([0])

    def test_multigraph_add_edges_from_four_tuple_misordered(self):
        """add_edges_from expects 4-tuples of the format (u, v, key, data_dict).

        Ensure 4-tuples of form (u, v, data_dict, key) raise exception.
        """
        G = nx.MultiGraph()
        with pytest.raises(TypeError):
            G.add_edges_from([(0, 1, {'color': 'red'}, 0)])

    def test_remove_edge(self):
        G = self.K3
        G.remove_edge(0, 1)
        assert G.adj == {0: {2: {0: {}}}, 1: {2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        with pytest.raises(nx.NetworkXError):
            G.remove_edge(-1, 0)
        with pytest.raises(nx.NetworkXError):
            G.remove_edge(0, 2, key=1)

    def test_remove_edges_from(self):
        G = self.K3.copy()
        G.remove_edges_from([(0, 1)])
        kd = {0: {}}
        assert G.adj == {0: {2: kd}, 1: {2: kd}, 2: {0: kd, 1: kd}}
        G.remove_edges_from([(0, 0)])
        self.K3.add_edge(0, 1)
        G = self.K3.copy()
        G.remove_edges_from(list(G.edges(data=True, keys=True)))
        assert G.adj == {0: {}, 1: {}, 2: {}}
        G = self.K3.copy()
        G.remove_edges_from(list(G.edges(data=False, keys=True)))
        assert G.adj == {0: {}, 1: {}, 2: {}}
        G = self.K3.copy()
        G.remove_edges_from(list(G.edges(data=False, keys=False)))
        assert G.adj == {0: {}, 1: {}, 2: {}}
        G = self.K3.copy()
        G.remove_edges_from([(0, 1, 0), (0, 2, 0, {}), (1, 2)])
        assert G.adj == {0: {1: {1: {}}}, 1: {0: {1: {}}}, 2: {}}

    def test_remove_multiedge(self):
        G = self.K3
        G.add_edge(0, 1, key='parallel edge')
        G.remove_edge(0, 1, key='parallel edge')
        assert G.adj == {0: {1: {0: {}}, 2: {0: {}}}, 1: {0: {0: {}}, 2: {0: {}}}, 2: {0: {0: {}}, 1: {0: {}}}}
        G.remove_edge(0, 1)
        kd = {0: {}}
        assert G.adj == {0: {2: kd}, 1: {2: kd}, 2: {0: kd, 1: kd}}
        with pytest.raises(nx.NetworkXError):
            G.remove_edge(-1, 0)