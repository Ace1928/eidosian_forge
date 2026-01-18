import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal, nodes_equal
class TestBipartiteProject:

    def test_path_projected_graph(self):
        G = nx.path_graph(4)
        P = bipartite.projected_graph(G, [1, 3])
        assert nodes_equal(list(P), [1, 3])
        assert edges_equal(list(P.edges()), [(1, 3)])
        P = bipartite.projected_graph(G, [0, 2])
        assert nodes_equal(list(P), [0, 2])
        assert edges_equal(list(P.edges()), [(0, 2)])
        G = nx.MultiGraph([(0, 1)])
        with pytest.raises(nx.NetworkXError, match='not defined for multigraphs'):
            bipartite.projected_graph(G, [0])

    def test_path_projected_properties_graph(self):
        G = nx.path_graph(4)
        G.add_node(1, name='one')
        G.add_node(2, name='two')
        P = bipartite.projected_graph(G, [1, 3])
        assert nodes_equal(list(P), [1, 3])
        assert edges_equal(list(P.edges()), [(1, 3)])
        assert P.nodes[1]['name'] == G.nodes[1]['name']
        P = bipartite.projected_graph(G, [0, 2])
        assert nodes_equal(list(P), [0, 2])
        assert edges_equal(list(P.edges()), [(0, 2)])
        assert P.nodes[2]['name'] == G.nodes[2]['name']

    def test_path_collaboration_projected_graph(self):
        G = nx.path_graph(4)
        P = bipartite.collaboration_weighted_projected_graph(G, [1, 3])
        assert nodes_equal(list(P), [1, 3])
        assert edges_equal(list(P.edges()), [(1, 3)])
        P[1][3]['weight'] = 1
        P = bipartite.collaboration_weighted_projected_graph(G, [0, 2])
        assert nodes_equal(list(P), [0, 2])
        assert edges_equal(list(P.edges()), [(0, 2)])
        P[0][2]['weight'] = 1

    def test_directed_path_collaboration_projected_graph(self):
        G = nx.DiGraph()
        nx.add_path(G, range(4))
        P = bipartite.collaboration_weighted_projected_graph(G, [1, 3])
        assert nodes_equal(list(P), [1, 3])
        assert edges_equal(list(P.edges()), [(1, 3)])
        P[1][3]['weight'] = 1
        P = bipartite.collaboration_weighted_projected_graph(G, [0, 2])
        assert nodes_equal(list(P), [0, 2])
        assert edges_equal(list(P.edges()), [(0, 2)])
        P[0][2]['weight'] = 1

    def test_path_weighted_projected_graph(self):
        G = nx.path_graph(4)
        with pytest.raises(nx.NetworkXAlgorithmError):
            bipartite.weighted_projected_graph(G, [1, 2, 3, 3])
        P = bipartite.weighted_projected_graph(G, [1, 3])
        assert nodes_equal(list(P), [1, 3])
        assert edges_equal(list(P.edges()), [(1, 3)])
        P[1][3]['weight'] = 1
        P = bipartite.weighted_projected_graph(G, [0, 2])
        assert nodes_equal(list(P), [0, 2])
        assert edges_equal(list(P.edges()), [(0, 2)])
        P[0][2]['weight'] = 1

    def test_digraph_weighted_projection(self):
        G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4)])
        P = bipartite.overlap_weighted_projected_graph(G, [1, 3])
        assert nx.get_edge_attributes(P, 'weight') == {(1, 3): 1.0}
        assert len(P) == 2

    def test_path_weighted_projected_directed_graph(self):
        G = nx.DiGraph()
        nx.add_path(G, range(4))
        P = bipartite.weighted_projected_graph(G, [1, 3])
        assert nodes_equal(list(P), [1, 3])
        assert edges_equal(list(P.edges()), [(1, 3)])
        P[1][3]['weight'] = 1
        P = bipartite.weighted_projected_graph(G, [0, 2])
        assert nodes_equal(list(P), [0, 2])
        assert edges_equal(list(P.edges()), [(0, 2)])
        P[0][2]['weight'] = 1

    def test_star_projected_graph(self):
        G = nx.star_graph(3)
        P = bipartite.projected_graph(G, [1, 2, 3])
        assert nodes_equal(list(P), [1, 2, 3])
        assert edges_equal(list(P.edges()), [(1, 2), (1, 3), (2, 3)])
        P = bipartite.weighted_projected_graph(G, [1, 2, 3])
        assert nodes_equal(list(P), [1, 2, 3])
        assert edges_equal(list(P.edges()), [(1, 2), (1, 3), (2, 3)])
        P = bipartite.projected_graph(G, [0])
        assert nodes_equal(list(P), [0])
        assert edges_equal(list(P.edges()), [])

    def test_project_multigraph(self):
        G = nx.Graph()
        G.add_edge('a', 1)
        G.add_edge('b', 1)
        G.add_edge('a', 2)
        G.add_edge('b', 2)
        P = bipartite.projected_graph(G, 'ab')
        assert edges_equal(list(P.edges()), [('a', 'b')])
        P = bipartite.weighted_projected_graph(G, 'ab')
        assert edges_equal(list(P.edges()), [('a', 'b')])
        P = bipartite.projected_graph(G, 'ab', multigraph=True)
        assert edges_equal(list(P.edges()), [('a', 'b'), ('a', 'b')])

    def test_project_collaboration(self):
        G = nx.Graph()
        G.add_edge('a', 1)
        G.add_edge('b', 1)
        G.add_edge('b', 2)
        G.add_edge('c', 2)
        G.add_edge('c', 3)
        G.add_edge('c', 4)
        G.add_edge('b', 4)
        P = bipartite.collaboration_weighted_projected_graph(G, 'abc')
        assert P['a']['b']['weight'] == 1
        assert P['b']['c']['weight'] == 2

    def test_directed_projection(self):
        G = nx.DiGraph()
        G.add_edge('A', 1)
        G.add_edge(1, 'B')
        G.add_edge('A', 2)
        G.add_edge('B', 2)
        P = bipartite.projected_graph(G, 'AB')
        assert edges_equal(list(P.edges()), [('A', 'B')])
        P = bipartite.weighted_projected_graph(G, 'AB')
        assert edges_equal(list(P.edges()), [('A', 'B')])
        assert P['A']['B']['weight'] == 1
        P = bipartite.projected_graph(G, 'AB', multigraph=True)
        assert edges_equal(list(P.edges()), [('A', 'B')])
        G = nx.DiGraph()
        G.add_edge('A', 1)
        G.add_edge(1, 'B')
        G.add_edge('A', 2)
        G.add_edge(2, 'B')
        P = bipartite.projected_graph(G, 'AB')
        assert edges_equal(list(P.edges()), [('A', 'B')])
        P = bipartite.weighted_projected_graph(G, 'AB')
        assert edges_equal(list(P.edges()), [('A', 'B')])
        assert P['A']['B']['weight'] == 2
        P = bipartite.projected_graph(G, 'AB', multigraph=True)
        assert edges_equal(list(P.edges()), [('A', 'B'), ('A', 'B')])