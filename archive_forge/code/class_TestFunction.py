import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
class TestFunction:

    def setup_method(self):
        self.G = nx.Graph({0: [1, 2, 3], 1: [1, 2, 0], 4: []}, name='Test')
        self.Gdegree = {0: 3, 1: 2, 2: 2, 3: 1, 4: 0}
        self.Gnodes = list(range(5))
        self.Gedges = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
        self.DG = nx.DiGraph({0: [1, 2, 3], 1: [1, 2, 0], 4: []})
        self.DGin_degree = {0: 1, 1: 2, 2: 2, 3: 1, 4: 0}
        self.DGout_degree = {0: 3, 1: 3, 2: 0, 3: 0, 4: 0}
        self.DGnodes = list(range(5))
        self.DGedges = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]

    def test_nodes(self):
        assert nodes_equal(self.G.nodes(), list(nx.nodes(self.G)))
        assert nodes_equal(self.DG.nodes(), list(nx.nodes(self.DG)))

    def test_edges(self):
        assert edges_equal(self.G.edges(), list(nx.edges(self.G)))
        assert sorted(self.DG.edges()) == sorted(nx.edges(self.DG))
        assert edges_equal(self.G.edges(nbunch=[0, 1, 3]), list(nx.edges(self.G, nbunch=[0, 1, 3])))
        assert sorted(self.DG.edges(nbunch=[0, 1, 3])) == sorted(nx.edges(self.DG, nbunch=[0, 1, 3]))

    def test_degree(self):
        assert edges_equal(self.G.degree(), list(nx.degree(self.G)))
        assert sorted(self.DG.degree()) == sorted(nx.degree(self.DG))
        assert edges_equal(self.G.degree(nbunch=[0, 1]), list(nx.degree(self.G, nbunch=[0, 1])))
        assert sorted(self.DG.degree(nbunch=[0, 1])) == sorted(nx.degree(self.DG, nbunch=[0, 1]))
        assert edges_equal(self.G.degree(weight='weight'), list(nx.degree(self.G, weight='weight')))
        assert sorted(self.DG.degree(weight='weight')) == sorted(nx.degree(self.DG, weight='weight'))

    def test_neighbors(self):
        assert list(self.G.neighbors(1)) == list(nx.neighbors(self.G, 1))
        assert list(self.DG.neighbors(1)) == list(nx.neighbors(self.DG, 1))

    def test_number_of_nodes(self):
        assert self.G.number_of_nodes() == nx.number_of_nodes(self.G)
        assert self.DG.number_of_nodes() == nx.number_of_nodes(self.DG)

    def test_number_of_edges(self):
        assert self.G.number_of_edges() == nx.number_of_edges(self.G)
        assert self.DG.number_of_edges() == nx.number_of_edges(self.DG)

    def test_is_directed(self):
        assert self.G.is_directed() == nx.is_directed(self.G)
        assert self.DG.is_directed() == nx.is_directed(self.DG)

    def test_add_star(self):
        G = self.G.copy()
        nlist = [12, 13, 14, 15]
        nx.add_star(G, nlist)
        assert edges_equal(G.edges(nlist), [(12, 13), (12, 14), (12, 15)])
        G = self.G.copy()
        nx.add_star(G, nlist, weight=2.0)
        assert edges_equal(G.edges(nlist, data=True), [(12, 13, {'weight': 2.0}), (12, 14, {'weight': 2.0}), (12, 15, {'weight': 2.0})])
        G = self.G.copy()
        nlist = [12]
        nx.add_star(G, nlist)
        assert nodes_equal(G, list(self.G) + nlist)
        G = self.G.copy()
        nlist = []
        nx.add_star(G, nlist)
        assert nodes_equal(G.nodes, self.Gnodes)
        assert edges_equal(G.edges, self.G.edges)

    def test_add_path(self):
        G = self.G.copy()
        nlist = [12, 13, 14, 15]
        nx.add_path(G, nlist)
        assert edges_equal(G.edges(nlist), [(12, 13), (13, 14), (14, 15)])
        G = self.G.copy()
        nx.add_path(G, nlist, weight=2.0)
        assert edges_equal(G.edges(nlist, data=True), [(12, 13, {'weight': 2.0}), (13, 14, {'weight': 2.0}), (14, 15, {'weight': 2.0})])
        G = self.G.copy()
        nlist = ['node']
        nx.add_path(G, nlist)
        assert edges_equal(G.edges(nlist), [])
        assert nodes_equal(G, list(self.G) + ['node'])
        G = self.G.copy()
        nlist = iter(['node'])
        nx.add_path(G, nlist)
        assert edges_equal(G.edges(['node']), [])
        assert nodes_equal(G, list(self.G) + ['node'])
        G = self.G.copy()
        nlist = [12]
        nx.add_path(G, nlist)
        assert edges_equal(G.edges(nlist), [])
        assert nodes_equal(G, list(self.G) + [12])
        G = self.G.copy()
        nlist = iter([12])
        nx.add_path(G, nlist)
        assert edges_equal(G.edges([12]), [])
        assert nodes_equal(G, list(self.G) + [12])
        G = self.G.copy()
        nlist = []
        nx.add_path(G, nlist)
        assert edges_equal(G.edges, self.G.edges)
        assert nodes_equal(G, list(self.G))
        G = self.G.copy()
        nlist = iter([])
        nx.add_path(G, nlist)
        assert edges_equal(G.edges, self.G.edges)
        assert nodes_equal(G, list(self.G))

    def test_add_cycle(self):
        G = self.G.copy()
        nlist = [12, 13, 14, 15]
        oklists = [[(12, 13), (12, 15), (13, 14), (14, 15)], [(12, 13), (13, 14), (14, 15), (15, 12)]]
        nx.add_cycle(G, nlist)
        assert sorted(G.edges(nlist)) in oklists
        G = self.G.copy()
        oklists = [[(12, 13, {'weight': 1.0}), (12, 15, {'weight': 1.0}), (13, 14, {'weight': 1.0}), (14, 15, {'weight': 1.0})], [(12, 13, {'weight': 1.0}), (13, 14, {'weight': 1.0}), (14, 15, {'weight': 1.0}), (15, 12, {'weight': 1.0})]]
        nx.add_cycle(G, nlist, weight=1.0)
        assert sorted(G.edges(nlist, data=True)) in oklists
        G = self.G.copy()
        nlist = [12]
        nx.add_cycle(G, nlist)
        assert nodes_equal(G, list(self.G) + nlist)
        G = self.G.copy()
        nlist = []
        nx.add_cycle(G, nlist)
        assert nodes_equal(G.nodes, self.Gnodes)
        assert edges_equal(G.edges, self.G.edges)

    def test_subgraph(self):
        assert self.G.subgraph([0, 1, 2, 4]).adj == nx.subgraph(self.G, [0, 1, 2, 4]).adj
        assert self.DG.subgraph([0, 1, 2, 4]).adj == nx.subgraph(self.DG, [0, 1, 2, 4]).adj
        assert self.G.subgraph([0, 1, 2, 4]).adj == nx.induced_subgraph(self.G, [0, 1, 2, 4]).adj
        assert self.DG.subgraph([0, 1, 2, 4]).adj == nx.induced_subgraph(self.DG, [0, 1, 2, 4]).adj
        H = nx.induced_subgraph(self.G.subgraph([0, 1, 2, 4]), [0, 1, 4])
        assert H._graph is not self.G
        assert H.adj == self.G.subgraph([0, 1, 4]).adj

    def test_edge_subgraph(self):
        assert self.G.edge_subgraph([(1, 2), (0, 3)]).adj == nx.edge_subgraph(self.G, [(1, 2), (0, 3)]).adj
        assert self.DG.edge_subgraph([(1, 2), (0, 3)]).adj == nx.edge_subgraph(self.DG, [(1, 2), (0, 3)]).adj

    def test_create_empty_copy(self):
        G = nx.create_empty_copy(self.G, with_data=False)
        assert nodes_equal(G, list(self.G))
        assert G.graph == {}
        assert G._node == {}.fromkeys(self.G.nodes(), {})
        assert G._adj == {}.fromkeys(self.G.nodes(), {})
        G = nx.create_empty_copy(self.G)
        assert nodes_equal(G, list(self.G))
        assert G.graph == self.G.graph
        assert G._node == self.G._node
        assert G._adj == {}.fromkeys(self.G.nodes(), {})

    def test_degree_histogram(self):
        assert nx.degree_histogram(self.G) == [1, 1, 1, 1, 1]

    def test_density(self):
        assert nx.density(self.G) == 0.5
        assert nx.density(self.DG) == 0.3
        G = nx.Graph()
        G.add_node(1)
        assert nx.density(G) == 0.0

    def test_density_selfloop(self):
        G = nx.Graph()
        G.add_edge(1, 1)
        assert nx.density(G) == 0.0
        G.add_edge(1, 2)
        assert nx.density(G) == 2.0

    def test_freeze(self):
        G = nx.freeze(self.G)
        assert G.frozen
        pytest.raises(nx.NetworkXError, G.add_node, 1)
        pytest.raises(nx.NetworkXError, G.add_nodes_from, [1])
        pytest.raises(nx.NetworkXError, G.remove_node, 1)
        pytest.raises(nx.NetworkXError, G.remove_nodes_from, [1])
        pytest.raises(nx.NetworkXError, G.add_edge, 1, 2)
        pytest.raises(nx.NetworkXError, G.add_edges_from, [(1, 2)])
        pytest.raises(nx.NetworkXError, G.remove_edge, 1, 2)
        pytest.raises(nx.NetworkXError, G.remove_edges_from, [(1, 2)])
        pytest.raises(nx.NetworkXError, G.clear_edges)
        pytest.raises(nx.NetworkXError, G.clear)

    def test_is_frozen(self):
        assert not nx.is_frozen(self.G)
        G = nx.freeze(self.G)
        assert G.frozen == nx.is_frozen(self.G)
        assert G.frozen

    def test_node_attributes_are_still_mutable_on_frozen_graph(self):
        G = nx.freeze(nx.path_graph(3))
        node = G.nodes[0]
        node['node_attribute'] = True
        assert node['node_attribute'] == True

    def test_edge_attributes_are_still_mutable_on_frozen_graph(self):
        G = nx.freeze(nx.path_graph(3))
        edge = G.edges[0, 1]
        edge['edge_attribute'] = True
        assert edge['edge_attribute'] == True

    def test_neighbors_complete_graph(self):
        graph = nx.complete_graph(100)
        pop = random.sample(list(graph), 1)
        nbors = list(nx.neighbors(graph, pop[0]))
        assert len(nbors) == len(graph) - 1
        graph = nx.path_graph(100)
        node = random.sample(list(graph), 1)[0]
        nbors = list(nx.neighbors(graph, node))
        if node != 0 and node != 99:
            assert len(nbors) == 2
        else:
            assert len(nbors) == 1
        graph = nx.star_graph(99)
        nbors = list(nx.neighbors(graph, 0))
        assert len(nbors) == 99

    def test_non_neighbors(self):
        graph = nx.complete_graph(100)
        pop = random.sample(list(graph), 1)
        nbors = list(nx.non_neighbors(graph, pop[0]))
        assert len(nbors) == 0
        graph = nx.path_graph(100)
        node = random.sample(list(graph), 1)[0]
        nbors = list(nx.non_neighbors(graph, node))
        if node != 0 and node != 99:
            assert len(nbors) == 97
        else:
            assert len(nbors) == 98
        graph = nx.star_graph(99)
        nbors = list(nx.non_neighbors(graph, 0))
        assert len(nbors) == 0
        graph = nx.Graph()
        graph.add_nodes_from(range(10))
        nbors = list(nx.non_neighbors(graph, 0))
        assert len(nbors) == 9

    def test_non_edges(self):
        graph = nx.complete_graph(5)
        nedges = list(nx.non_edges(graph))
        assert len(nedges) == 0
        graph = nx.path_graph(4)
        expected = [(0, 2), (0, 3), (1, 3)]
        nedges = list(nx.non_edges(graph))
        for u, v in expected:
            assert (u, v) in nedges or (v, u) in nedges
        graph = nx.star_graph(4)
        expected = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        nedges = list(nx.non_edges(graph))
        for u, v in expected:
            assert (u, v) in nedges or (v, u) in nedges
        graph = nx.DiGraph()
        graph.add_edges_from([(0, 2), (2, 0), (2, 1)])
        expected = [(0, 1), (1, 0), (1, 2)]
        nedges = list(nx.non_edges(graph))
        for e in expected:
            assert e in nedges

    def test_is_weighted(self):
        G = nx.Graph()
        assert not nx.is_weighted(G)
        G = nx.path_graph(4)
        assert not nx.is_weighted(G)
        assert not nx.is_weighted(G, (2, 3))
        G.add_node(4)
        G.add_edge(3, 4, weight=4)
        assert not nx.is_weighted(G)
        assert nx.is_weighted(G, (3, 4))
        G = nx.DiGraph()
        G.add_weighted_edges_from([('0', '3', 3), ('0', '1', -5), ('1', '0', -5), ('0', '2', 2), ('1', '2', 4), ('2', '3', 1)])
        assert nx.is_weighted(G)
        assert nx.is_weighted(G, ('1', '0'))
        G = G.to_undirected()
        assert nx.is_weighted(G)
        assert nx.is_weighted(G, ('1', '0'))
        pytest.raises(nx.NetworkXError, nx.is_weighted, G, (1, 2))

    def test_is_negatively_weighted(self):
        G = nx.Graph()
        assert not nx.is_negatively_weighted(G)
        G.add_node(1)
        G.add_nodes_from([2, 3, 4, 5])
        assert not nx.is_negatively_weighted(G)
        G.add_edge(1, 2, weight=4)
        assert not nx.is_negatively_weighted(G, (1, 2))
        G.add_edges_from([(1, 3), (2, 4), (2, 6)])
        G[1][3]['color'] = 'blue'
        assert not nx.is_negatively_weighted(G)
        assert not nx.is_negatively_weighted(G, (1, 3))
        G[2][4]['weight'] = -2
        assert nx.is_negatively_weighted(G, (2, 4))
        assert nx.is_negatively_weighted(G)
        G = nx.DiGraph()
        G.add_weighted_edges_from([('0', '3', 3), ('0', '1', -5), ('1', '0', -2), ('0', '2', 2), ('1', '2', -3), ('2', '3', 1)])
        assert nx.is_negatively_weighted(G)
        assert not nx.is_negatively_weighted(G, ('0', '3'))
        assert nx.is_negatively_weighted(G, ('1', '0'))
        pytest.raises(nx.NetworkXError, nx.is_negatively_weighted, G, (1, 4))