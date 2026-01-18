import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
class TestGeneratorInverseLine:

    def test_example(self):
        G = nx.Graph()
        G_edges = [[1, 2], [1, 3], [1, 4], [1, 5], [2, 3], [2, 5], [2, 6], [2, 7], [3, 4], [3, 5], [6, 7], [6, 8], [7, 8]]
        G.add_edges_from(G_edges)
        H = nx.inverse_line_graph(G)
        solution = nx.Graph()
        solution_edges = [('a', 'b'), ('a', 'c'), ('a', 'd'), ('a', 'e'), ('c', 'd'), ('e', 'f'), ('e', 'g'), ('f', 'g')]
        solution.add_edges_from(solution_edges)
        assert nx.is_isomorphic(H, solution)

    def test_example_2(self):
        G = nx.Graph()
        G_edges = [[1, 2], [1, 3], [2, 3], [3, 4], [3, 5], [4, 5]]
        G.add_edges_from(G_edges)
        H = nx.inverse_line_graph(G)
        solution = nx.Graph()
        solution_edges = [('a', 'c'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('d', 'f')]
        solution.add_edges_from(solution_edges)
        assert nx.is_isomorphic(H, solution)

    def test_pair(self):
        G = nx.path_graph(2)
        H = nx.inverse_line_graph(G)
        solution = nx.path_graph(3)
        assert nx.is_isomorphic(H, solution)

    def test_line(self):
        G = nx.path_graph(5)
        solution = nx.path_graph(6)
        H = nx.inverse_line_graph(G)
        assert nx.is_isomorphic(H, solution)

    def test_triangle_graph(self):
        G = nx.complete_graph(3)
        H = nx.inverse_line_graph(G)
        alternative_solution = nx.Graph()
        alternative_solution.add_edges_from([[0, 1], [0, 2], [0, 3]])
        assert nx.is_isomorphic(H, G) or nx.is_isomorphic(H, alternative_solution)

    def test_cycle(self):
        G = nx.cycle_graph(5)
        H = nx.inverse_line_graph(G)
        assert nx.is_isomorphic(H, G)

    def test_empty(self):
        G = nx.Graph()
        H = nx.inverse_line_graph(G)
        assert nx.is_isomorphic(H, nx.complete_graph(1))

    def test_K1(self):
        G = nx.complete_graph(1)
        H = nx.inverse_line_graph(G)
        solution = nx.path_graph(2)
        assert nx.is_isomorphic(H, solution)

    def test_edgeless_graph(self):
        G = nx.empty_graph(5)
        with pytest.raises(nx.NetworkXError, match='edgeless graph'):
            nx.inverse_line_graph(G)

    def test_selfloops_error(self):
        G = nx.cycle_graph(4)
        G.add_edge(0, 0)
        pytest.raises(nx.NetworkXError, nx.inverse_line_graph, G)

    def test_non_line_graphs(self):
        claw = nx.star_graph(3)
        pytest.raises(nx.NetworkXError, nx.inverse_line_graph, claw)
        wheel = nx.wheel_graph(6)
        pytest.raises(nx.NetworkXError, nx.inverse_line_graph, wheel)
        K5m = nx.complete_graph(5)
        K5m.remove_edge(0, 1)
        pytest.raises(nx.NetworkXError, nx.inverse_line_graph, K5m)
        G = nx.compose(nx.path_graph(2), nx.complete_bipartite_graph(2, 3))
        pytest.raises(nx.NetworkXError, nx.inverse_line_graph, G)
        G = nx.diamond_graph()
        G.add_edges_from([(4, 0), (5, 3)])
        pytest.raises(nx.NetworkXError, nx.inverse_line_graph, G)
        G.add_edge(4, 5)
        pytest.raises(nx.NetworkXError, nx.inverse_line_graph, G)
        G = nx.diamond_graph()
        G.add_edges_from([(4, 0), (4, 3)])
        pytest.raises(nx.NetworkXError, nx.inverse_line_graph, G)
        G = nx.diamond_graph()
        G.add_edges_from([(4, 0), (4, 1), (4, 2), (5, 3)])
        pytest.raises(nx.NetworkXError, nx.inverse_line_graph, G)
        G.add_edges_from([(5, 1), (5, 2)])
        pytest.raises(nx.NetworkXError, nx.inverse_line_graph, G)
        G = nx.diamond_graph()
        G.add_edges_from([(4, 0), (4, 1), (5, 2), (5, 3)])
        pytest.raises(nx.NetworkXError, nx.inverse_line_graph, G)

    def test_wrong_graph_type(self):
        G = nx.DiGraph()
        G_edges = [[0, 1], [0, 2], [0, 3]]
        G.add_edges_from(G_edges)
        pytest.raises(nx.NetworkXNotImplemented, nx.inverse_line_graph, G)
        G = nx.MultiGraph()
        G_edges = [[0, 1], [0, 2], [0, 3]]
        G.add_edges_from(G_edges)
        pytest.raises(nx.NetworkXNotImplemented, nx.inverse_line_graph, G)

    def test_line_inverse_line_complete(self):
        G = nx.complete_graph(10)
        H = nx.line_graph(G)
        J = nx.inverse_line_graph(H)
        assert nx.is_isomorphic(G, J)

    def test_line_inverse_line_path(self):
        G = nx.path_graph(10)
        H = nx.line_graph(G)
        J = nx.inverse_line_graph(H)
        assert nx.is_isomorphic(G, J)

    def test_line_inverse_line_hypercube(self):
        G = nx.hypercube_graph(5)
        H = nx.line_graph(G)
        J = nx.inverse_line_graph(H)
        assert nx.is_isomorphic(G, J)

    def test_line_inverse_line_cycle(self):
        G = nx.cycle_graph(10)
        H = nx.line_graph(G)
        J = nx.inverse_line_graph(H)
        assert nx.is_isomorphic(G, J)

    def test_line_inverse_line_star(self):
        G = nx.star_graph(20)
        H = nx.line_graph(G)
        J = nx.inverse_line_graph(H)
        assert nx.is_isomorphic(G, J)

    def test_line_inverse_line_multipartite(self):
        G = nx.complete_multipartite_graph(3, 4, 5)
        H = nx.line_graph(G)
        J = nx.inverse_line_graph(H)
        assert nx.is_isomorphic(G, J)

    def test_line_inverse_line_dgm(self):
        G = nx.dorogovtsev_goltsev_mendes_graph(4)
        H = nx.line_graph(G)
        J = nx.inverse_line_graph(H)
        assert nx.is_isomorphic(G, J)

    def test_line_different_node_types(self):
        G = nx.path_graph([1, 2, 3, 'a', 'b', 'c'])
        H = nx.line_graph(G)
        J = nx.inverse_line_graph(H)
        assert nx.is_isomorphic(G, J)