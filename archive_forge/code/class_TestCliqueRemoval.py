import networkx as nx
from networkx.algorithms.approximation import (
class TestCliqueRemoval:
    """Unit tests for the
    :func:`~networkx.algorithms.approximation.clique_removal` function.

    """

    def test_trivial_graph(self):
        G = nx.trivial_graph()
        independent_set, cliques = clique_removal(G)
        assert is_independent_set(G, independent_set)
        assert all((is_clique(G, clique) for clique in cliques))
        assert all((len(clique) == 1 for clique in cliques))

    def test_complete_graph(self):
        G = nx.complete_graph(10)
        independent_set, cliques = clique_removal(G)
        assert is_independent_set(G, independent_set)
        assert all((is_clique(G, clique) for clique in cliques))

    def test_barbell_graph(self):
        G = nx.barbell_graph(10, 5)
        independent_set, cliques = clique_removal(G)
        assert is_independent_set(G, independent_set)
        assert all((is_clique(G, clique) for clique in cliques))